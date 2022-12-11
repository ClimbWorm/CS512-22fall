import torch
import torch.nn as nn
from GRU import GRUCell,GRUHelper


<<<<<<< HEAD


class Encoder(nn.Module,GRUHelper):

    def __init__(self, adj_mx, model_kwargs):
        nn.Module.__init__(self)
        GRUHelper.__init__(self, adj_mx, model_kwargs)

        self.input_dim = int(self.arguments.get_model().get('input_dim', 1))#todo set to 2?
        self.seq_len = int(self.arguments.get_model().get('seq_len',12))  # for the encoder 12
        self.dcgru_layers = nn.ModuleList(
            [GRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type) for _ in range(self.num_rnn_layers)])


class DecoderModel(nn.Module, GRUHelper):
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

=======
class Encoder(nn.Module):
    pass
>>>>>>> 40786f99e872c9e5c9cdcc032bc42f87c6f524f4


class Decoder(nn.Module):
    pass


class DCRNN(nn.Module):
    pass
