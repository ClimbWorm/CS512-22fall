import torch
import torch.nn as nn
from GRU import GRUCell,ArgsReader
from typing import List, Tuple, Dict


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
        self.num_rnn_layers = int(model_kwargs.get_model().get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get_model().get('rnn_units'))
        self.hidden_state_size = self.num_nodes * self.rnn_units



class Encoder(nn.Module,GRUHelper):

    def __init__(self, adj_mx, model_kwargs):
        nn.Module.__init__(self)
        GRUHelper.__init__(self, adj_mx, model_kwargs)

        self.input_dim = int(self.arguments.get_model().get('input_dim', 1))#todo set to 2?
        self.seq_len = int(self.arguments.get_model().get('seq_len',12))  # for the encoder 12
        self.dcgru_layers = nn.ModuleList(
            [GRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])


class Decoder(nn.Module, GRUHelper):
    def __init__(self, adj_mx, model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        GRUHelper.__init__(self, adj_mx, model_kwargs)

        self.output_dim = int(self.arguments.get_model().get('output_dim', 1))
        self.horizon = int(self.arguments.get_model().get('horizon', 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [GRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])


class DCRNN(nn.Module):
    pass
