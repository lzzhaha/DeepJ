import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from constants import *
from util import *
import numpy as np

class DeepJ(nn.Module):
    def __init__(self, input_size=512, encoder_size=512, decoder_size=512, latent_size=512):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.embd = nn.Embedding(NUM_ACTIONS, input_size)
        self.encoder = EncoderRNN(input_size, encoder_size, latent_size, 4)
        self.decoder = DecoderRNN(self.embd, input_size, latent_size, decoder_size, NUM_ACTIONS, 1)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        # Project to dense representation
        x = self.embd(x)
        # Encoder output is the latent vector
        mean, logvar, _ = self.encoder(x, hidden)
        std = torch.exp(0.5 * logvar)
        
        # Generate random latent vector
        z = Variable(torch.randn([batch_size, self.latent_size]))
        z = z.type(type(x.data))

        if x.is_cuda:
            z = z.cuda()

        z = z * std + mean

        decoder_output, _ = self.decoder(x[:, :-1], z)
        return decoder_output, mean, logvar


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.rnn = nn.GRU(input_size, hidden_size // 2, num_layers, batch_first=True, bidirectional=True)
        self.latent_projection = nn.Linear(hidden_size * num_layers, latent_size * 2)

    def forward(self, x, hidden=None):
        """
        Takes in a sequence of inputs and encodes them into mean and log variance.
        Return: (Mean, SD, Hidden)
        """
        _, hidden = self.rnn(x, hidden)
        
        # Project hidden state to latent vector
        x = torch.cat(list(hidden), dim=1)
        x = self.latent_projection(x)

        return x[:, :self.latent_size], x[:, self.latent_size:], hidden


class DecoderRNN(nn.Module):
    def __init__(self, embd, input_size, latent_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.branch_factor = 128

        self.latent_projection = nn.Linear(latent_size, hidden_size * num_layers)
        self.decoder1 = nn.GRU(1, hidden_size, num_layers, batch_first=True)
        self.decoder2 = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        
        # Tie output embedding with input embd weights to improve regularization
        # https://arxiv.org/abs/1608.05859 https://arxiv.org/abs/1611.01462
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = embd.weight

        # self.dropout = nn.Dropout(0.5)

    def project_latent(self, latent):
        latent = self.latent_projection(latent)
        latent = latent.view(-1, self.num_layers, self.hidden_size)
        latent = latent.permute(1, 0, 2).contiguous()
        return latent

    def forward(self, inputs, latent=None, hidden=None):
        batch_size, seq_len, num_features = inputs.size()
        assert (latent is None and hidden is not None) or (hidden is None and latent is not None)

        if seq_len == 1:
            # Project the latent vector to a size consumable by the GRU's memory
            if hidden is None:
                hidden = (0, self.project_latent(latent), None)
            
            index, h1, h2 = hidden

            if index == 0:
                # A dummy tensor to feed zeros into the first GRU input
                length_input = Variable(torch.zeros(batch_size, 1, 1))
                length_input = length_input.type(type(inputs.data))

                if length_input.is_cuda:
                    length_input = length_input.cuda()

                h2, h1 = self.decoder1(length_input, h1)
            
            x, h2 = self.decoder2(inputs, h2)
            x = self.output(x)
            return x, ((index + 1) % self.branch_factor, h1, h2)
        else:
            # Project the latent vector to a size consumable by the GRU's memory
            if hidden is None:
                hidden = (self.project_latent(latent), None)

            assert seq_len % self.branch_factor == 0
            
            h1, h2 = hidden
            # A dummy tensor to feed zeros into the first GRU input
            length_input = Variable(torch.zeros(batch_size, seq_len // self.branch_factor, 1))
            length_input = length_input.type(type(inputs.data))

            if length_input.is_cuda:
                length_input = length_input.cuda()

            coarse_features, h1 = self.decoder1(length_input, h1)
            coarse_features = coarse_features.contiguous().view(-1, self.hidden_size)

            inputs = inputs.contiguous().view(-1, self.branch_factor, num_features)
            x, h2 = self.decoder2(inputs, coarse_features.unsqueeze(0))
            x = x.contiguous().view(batch_size, seq_len, -1)
            x = self.output(x)
            return x, (h1, h2)
