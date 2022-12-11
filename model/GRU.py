import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
import os
import json
import pprint
import scipy.sparse as sp
from scripts import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParamsHelper:
    """
    weights and biases in GRU units
    """

    def __init__(self, rnn_network: nn.Module, layer_name: str):
        """
        :param rnn_network: a nn.Module object
        :param layer_name: 'fc': fully connected layer  or "gc":graph convolutional layer
        """
        self._rnn_network = rnn_network
        self._name = layer_name
        self._weights_dict = {}
        self._biases_dict = {}

    def get_weights(self, shape: Tuple[int, int]):
        """
        get the weights in neural network, same shape same weights (e.g same U,V matrix)
        :param shape: shape of weight matrix: (input_size, output_size)
        :return: weights dict
        """
        if shape not in self._weights_dict:
            nn_weights = nn.Parameter(torch.empty(*shape, device=device))
            nn.init.xavier_normal_(nn_weights)  # change in-place
            self._weights_dict[shape] = nn_weights
            # mark the weights as trainable
            self._rnn_network.register_parameter(name='{}_weight_{}'.format(self._name, str(shape)),
                                                 param=nn_weights)
        return self._weights_dict[shape]

    def get_biases(self, length: int, init_val: float = 0.0):
        """
        get the weights in neural network, same shape same weights (e.g same U,V matrix)
        :param length: length of bias: output_size
        :param init_val: value initialized for biases
        :return: weights dict
        """
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length, device=device, dtype=torch.float32))
            nn.init.constant_(tensor=biases, val=init_val)
            self._rnn_network.register_parameter(name='{}_biases_{}'.format(self._name, str(length)), param=biases)
            self._biases_dict[length] = biases
        return self._biases_dict[length]


class ArgsReader:
    """
    config file
    """
    config_path = ""
    model_config = dict()
    train_config = dict()

    # todo data_config not include

    def __init__(self, config_path: str) -> None:
        if (not os.path.exists(config_path)):
            raise Exception("Unknown config path!")

        model_config_file = os.path.join(config_path, "args_model.json")
        print('Reading model config from: ' + model_config_file)
        with open(model_config_file, "r") as cfg_f:
            self.model_config = json.loads(cfg_f.read())
            pprint.pprint(self.model_config)

        train_config_file = os.path.join(config_path, "args_train.json")
        print('Reading train config from: ' + train_config_file)
        with open(train_config_file, "r") as cfg_f:
            self.train_config = json.loads(cfg_f.read())
            pprint.pprint(self.train_config)

    def get_model(self) -> Dict:
        return self.model_config['model']

    def get_train(self) -> Dict:
        return self.train_config['train']


class GRUCell(nn.Module):  # diffusion convolution GRU
    def __init__(self, num_units: int, adj_mx: List[List[float]], max_diffusion_step: int, num_nodes: int,
                 nonlinearity: str = 'tanh', filter_type: str = "laplacian",
                 use_gc_for_ru: bool = True):
        """
        :param num_units: units in the hidden state
        :param adj_mx: adjacency matrix of sensors
        :param max_diffusion_step: maximum steps allowed for diffusion process
        :param num_nodes: number of total sensors
        :param nonlinearity: activation function
        :param filter_type: "laplacian", "random_walk", "dual_random_walk". applied on adj_mx
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        nn.Module.__init__(self)
        self._num_units = num_units
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu

        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        supports = []

        if filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self._fc_params = ParamsHelper(self, 'fc')
        self._gc_params = ParamsHelper(self, 'gc')

    @staticmethod
    def _build_sparse_matrix(L: sp.csc_matrix) -> torch.sparse_coo_tensor:
        """
        :param L: single transformed adj_max using "laplacian"/"random_walk"/"dual_random_walk"
        :return: a new sparse tensor in the format can be used by PyTorch
        """
        # stores the row, column, and data values of non-zero elements in separate arrays
        L = L.tocoo()
        # creates a list of coordinates for each non-zero element in the matrix
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering ,which is the default ordering used by PyTorch's sparse tensors:
        # torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        # creates a new sparse tensor in the format used by PyTorch.
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L

    def forward(self, inputs, hx):
        """
        Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: X (Batch, num_nodes * input_dim) todo 这个和 (epoch_size, input_length, num_nodes, input_dim)不符？是需要输入forward前再对Xreshape吗
        :param hx: hidden state (Batch, num_nodes * rnn_units)
        :return A new hidden state: `2-D` tensor with shape `(Batch, num_nodes * rnn_units)`
        """
        # *2 because we will split  value into 2 parts, reset gate and update gate, they have the same structure
        # note: output_size of reset/update gate = #rnn_units in hidden state
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._gc
        else:
            fn = self._fc

        # return shape (batch_size,num_nodes,output_size/rnn_units)
        value = torch.sigmoid(
            fn(inputs=inputs, state=hx, output_size=output_size, bias_init=1.0))  # todo _fc里面已经自带sigmoid
        value = torch.reshape(value, (-1, self._num_nodes, output_size)) # original

        # split into reset and update gate by the last dimension, namely output_size todo 直接这样分开的话r和u的结构能一样吗？
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        # reshape to let the shape of outputs from reset and update gate match with the hidden state hx
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        # computes the values of the candidate new state using graph convolution,
        # and applies an activation function if one is specified.
        c = self._gc(inputs=inputs, state=r * hx, output_size=self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        # new_state = u * hx + (1.0 - u) * c #original code
        # calculate hidden state at next time step
        new_state = (1.0 - u) * hx + u * c

        return new_state

    def _fc(self, inputs, state, output_size, bias_init=0.0):
        """
        fully connected layer used in reset/update gate or candidate new state generation
        :param inputs: X (Batch, num_nodes * input_dim)
        :param state: r*hx or hx (Batch, num_nodes * rnn_units)
        :param output_size: same size as hidden state (Batch, num_nodes * rnn_units)
        :param bias_init: initialize bias
        :return: output from fully connected layer (batch_size,self._num_nodes,output_size/rnn_units)
        """
        batch_size = inputs.shape[0]
        # let input_dim be the 2nd dimension
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        # let #rnn_units be the 2nd dimension
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))

        inputs_and_state = torch.cat([inputs, state],
                                     dim=-1)  # size: (batch_size * num_nodes, input_dim + rnn_units)
        input_size = inputs_and_state.shape[-1]  # value: input_dim + rnn_units

        weights = self._fc_params.get_weights((input_size, output_size))
        # (batch_size * num_nodes, input_dim + rnn_units) * (input_dim + rnn_units, output_size)
        # = (batch_size * num_nodes, output_size)
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))  # todo why sigmoid here but not in _gc?
        biases = self._fc_params.get_biases(output_size, bias_init)
        value += biases
        # value = torch.reshape(value, (batch_size, self._num_nodes * output_size))
        return value

    @staticmethod
    def _concat(x, x_):
        """
        purpose: do not change the value of x_ outside the function
        :param x:
        :param x_:
        :return:
        """
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gc(self, inputs, state, output_size, bias_init=0.0):
        """
        graph convolutional layer used in reset/update gate or candidate new state generation
        :param inputs: X (Batch, num_nodes * input_dim)
        :param state: r*hx or hx (Batch, num_nodes * rnn_units)
        :param output_size: same size as hidden state (Batch, num_nodes * rnn_units)
        :param bias_init: initialize bias
        :return: (batch_size, self._num_nodes, output_size/rnn_units)
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        # let input_dim be the 3rd dimension
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))  # todo why reshape differently with fc?
        # let #rnn_units be the 3rd dimension
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))

        inputs_and_state = torch.cat([inputs, state], dim=2)  # size: (batch_size, num_nodes, input_dim + rnn_units)
        input_size = inputs_and_state.size(2)  # value: input_dim + rnn_units

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, input_dim + rnn_units, batch_size)
        x0 = torch.reshape(x0, (self._num_nodes, input_size * batch_size))  # (num_nodes, input_size * batch_size)
        x = torch.unsqueeze(x0, 0)  # (-1, num_nodes, input_size * batch_size)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # Performs a matrix multiplication of the sparse matrix mat1 and the (sparse or dense) matrix mat2.
                # (num_nodes,num_nodes) * (num_nodes, input_size * batch_size) = (num_nodes,input_size * batch_size)
                x1 = torch.sparse.mm(support, x0)  # after unsqueeze(0): (-1, num_nodes,input_size * batch_size)
                # stack one more matrix x1 on x for each loop
                x = self._concat(x, x1)  # (-1, num_nodes,input_size * batch_size)

                # diffusion
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0  # (num_nodes, input_size * batch_size)todo represent?
                    # stack one more matrix x1 on x for each loop
                    x = self._concat(x, x2)  # (-1, num_nodes,input_size * batch_size)
                    x1, x0 = x2, x1

        # number of stacked matrices, +1: Adds for x itself.
        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(x, (num_matrices, self._num_nodes, input_size, batch_size))
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        # final matrix after convolutional layer
        x = torch.reshape(x, (batch_size * self._num_nodes, input_size * num_matrices))

        weights = self._gc_params.get_weights((input_size * num_matrices, output_size))
        # (batch_size * num_nodes, input_size * num_matrices) * (input_size * num_matrices, output_size)
        # = (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, weights)

        biases = self._gc_params.get_biases(output_size, bias_init)
        x += biases
        # Reshape res to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        x = torch.reshape(x, (batch_size, self._num_nodes * output_size))
        return x
