import torch
import torch.nn as nn
from typing import List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParamsHelper:
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
        return self._biases_dict[length]


class GRUCell(nn.Module):

    def __init__(self):
        super(GRUCell, self).__init__()
