import torch
import torch.nn as nn
import numpy as np
from GRU import GRUCell, ArgsReader
from typing import List, Tuple, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# sequence of gru units
class GRUHelper:
    """
    parameters for GRUCell
    """

    def __init__(self, adj_mx: List[List[float]], model_kwargs: ArgsReader):
        self.arguments = model_kwargs
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(model_kwargs.get_model().get('max_diffusion_step', 2))
        self.cl_decay_steps = int(model_kwargs.get_model().get('cl_decay_steps', 1000))
        self.filter_type = model_kwargs.get_model().get('filter_type', 'laplacian')
        self.num_nodes = int(model_kwargs.get_model().get('num_nodes', 1))
        self.num_rnn_layers = int(model_kwargs.get_model().get('num_rnn_layers', 1))  # number of GRUCell unit
        self.rnn_units = int(model_kwargs.get_model().get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class Encoder(nn.Module, GRUHelper):
    def __init__(self, adj_mx, model_kwargs: ArgsReader):
        nn.Module.__init__(self)
        GRUHelper.__init__(self, adj_mx, model_kwargs)

        self.input_dim = int(self.arguments.get_model().get('input_dim', 1))  # todo set to 2? [speed,timeinday]
        # the number of history time step we based on to make inference
        self.seq_len = int(self.arguments.get_model().get('seq_len', 12))  # for the encoder 12
        self.dcgru_layers = nn.ModuleList(
            [GRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                     filter_type=self.filter_type) for _ in
             range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.
        :param inputs: hx at time step t, shape (batch_size, self.num_nodes * self.input_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.hidden_state_size)
                 next_hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """

        # only useful at the first encoder
        batch_size, _ = inputs.size()
        if hidden_state is None:  # when it's the first time into GRUCell
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)


class Decoder(nn.Module, GRUHelper):
    def __init__(self, adj_mx, model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        GRUHelper.__init__(self, adj_mx, model_kwargs)

        self.output_dim = int(self.arguments.get_model().get('output_dim', 1))
        # for the decoder, number of future steps we want to predict
        self.horizon = int(self.arguments.get_model().get('horizon', 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [GRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                     filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.
        :param inputs: the last output of the Encoder or previous decoder, shape (batch_size, self.num_nodes * self.output_dim)
        :param hidden_state: (num_layers, batch_size, self.hidden_state_size)
               optional, zeros if not provided
        :return: output: # shape (batch_size, self.num_nodes * self.output_dim)
                 next_hidden_state # shape (num_layers, batch_size, self.hidden_state_size)
                 (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # (num_layers, batch_size, self.hidden_state_size)

        # only useful when at the last decoder
        projected = self.projection_layer(output.view(-1, self.rnn_units))  # (-1,self.output_dim)
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNN(nn.Module, GRUHelper):
    def __init__(self, adj_mx, model_kwargs):
        nn.Module.__init__(self)
        GRUHelper.__init__(self, adj_mx, model_kwargs)

        self.encoder = Encoder(adj_mx, model_kwargs)
        self.decoder = Decoder(adj_mx, model_kwargs)

        # the following 2 params control the Curriculum learning
        # It is a machine learning technique that involves training a model on a sequence of
        # increasingly difficult tasks, in order to improve its performance.

        # the rate at which the difficulty of the tasks increases during training, controls the rate of decay
        # in the exponential function, and it determines how quickly the sampling threshold decreases from 1 to 0 as
        # the number of batches increases.
        self.cl_decay_steps = int(self.arguments.get_model().get('cl_decay_steps', 1000))
        self.use_curriculum_learning = bool(model_kwargs.get('use_curriculum_learning', False))

    def _calc_sampling_threshold(self, batches_at):
        """
        an exponential decay function determines the probability that the model will sample a difficult task for training, instead of an easy task.
        :param batches_at: number of batches already processed, batches seen till now
        :return: a value between 0 and 1
        """
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_at / self.cl_decay_steps))

    def encoder_forward(self, inputs):
        """
        encoder forward pass for t time steps
        :param inputs: generated by generate_data function, shape (seq_len, batch_size, num_sensor * input_dim)
        :return: encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.encoder.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
        return encoder_hidden_state

    def decoder_forward(self, encoder_hidden_state, labels=None, batches_at=None):
        """
        Decoder forward pass
        :param encoder_hidden_state: last hidden state from encoder(num_layers, batch_size, self.hidden_state_size)
        :param labels: the correct output values for the model(self.horizon, batch_size, self.num_nodes * self.output_dim) [optional, not exist for inference]
        :param batches_at: global step [optional, not exist for inference]
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)

        # define the initial input to the decoder, and it is used to start the decoder forward pass
        start_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder.output_dim), device=device)

        # the last hidden state of encoder is the last hidden state of decoder
        decoder_hidden_state = encoder_hidden_state
        decoder_input = start_symbol

        outputs = []

        for t in range(self.decoder.horizon):
            decoder_output, decoder_hidden_state = self.decoder(decoder_input,decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            # RNN can be in training or evaluating mode by calling train() or eval() respectively
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._calc_sampling_threshold(batches_at):
                    decoder_input = labels[t]

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_at=None):
        """
        seq2seq forward pass
        :param inputs: generated by generate_data function, shape (seq_len, batch_size, num_sensor * input_dim)
        :param labels: shape (horizon, batch_size, num_sensor * output)
        :param batches_at: batches seen till now
        :return: output: (self.horizon, batch_size, self.num_nodes * self.output_dim)
        """
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_at)

        return outputs
