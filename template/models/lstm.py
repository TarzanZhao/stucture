import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
import torch.optim as optim
from template.arguments import get_args
from template.collate_fns import fulltensor_collate_fn
import numpy as np
from template.models.modules import NaiveFE, TimeEmbedding
####################model structure#########################


class LSTMmodel(nn.Module):
    """
    LSTM encoder.

    Args:
        config: 
    """

    def __init__(self):
        super(LSTMmodel, self).__init__()
        self.args = get_args()
        self.lstm = nn.LSTM(self.args.lstm_d_hidden,
                            self.args.lstm_d_hidden, self.args.lstm_N, batch_first=True)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.args.lstm_n_layers, batch_size, self.args.lstm_d_hidden, device=device)

    def initGate(self, batch_size, device):
        return torch.zeros(self.args.lstm_n_layers, batch_size, self.args.lstm_d_hidden, device=device)

    def forward(self, x, hiddens=None):  # x: (bsz, seqlen, lstm_d_hidden)
        """
        Args: input_ids (LongTensor): tokens in the source language of shape `(batch, src_len)`
        """
        bsz = x.size(0)
        if hiddens is None:
            h_0 = self.initHidden(bsz, device=x.device)
            c_0 = self.initGate(bsz, device=x.device)
            hiddens = (h_0, c_0)
        out, hiddens = self.lstm(x, hiddens)
        return out, hiddens

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_block=2, dropout=0.3):
        super(MLP, self).__init__() 
        self.dropout = dropout
        self.fc1 = nn.Sequential(nn.Linear(num_inputs, num_hidden),
                                nn.ReLU())
        self.res_block = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for i in range(num_block)])
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        for block in self.res_block:
            x = F.relu(x + block(x))
            x = F.dropout(x, p=self.dropout)
        x = self.fc2(x)
        return x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        if hidden is None:
            output, hidden = self.gru(x)
        else:
            output, hidden = self.gru(x, hidden)
        return output, hidden

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

class RNN_v2(nn.Module):
    def __init__(self, device, examination_size, examination_proj_size,
                insulin_size, insulin_proj_size,
                sugar_size, sugar_proj_size,
                day_proj_size,
                time_proj_size,
                drug_size, drug_proj_size,
                hidden_size, num_layers, output_size):
        super(RNN_v2, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = insulin_proj_size + sugar_proj_size + drug_proj_size + time_proj_size + day_proj_size+examination_proj_size

        self.patient2hidden = MLP(examination_size, 256, examination_proj_size, num_block=2)
        self.insulin_proj = nn.Linear(insulin_size, insulin_proj_size)
        self.sugar_proj = nn.Linear(sugar_size, sugar_proj_size)
        self.day_embedding = nn.Embedding(50, day_proj_size, padding_idx=0)
        self.time_embedding = nn.Embedding(8, time_proj_size, padding_idx=-1)
        self.drug_proj = nn.Linear(drug_size, drug_proj_size)
        self.encoder = EncoderRNN(self.input_size, hidden_size, num_layers)
        self.out_proj = MLP(hidden_size+self.input_size, hidden_size, output_size, 3)
    
    def collate_fn(self, samples):
        return fulltensor_collate_fn(samples)

    def get_loss(self, data):
        vec = data["examination"].to(self.device)
        insulin = self.insulin_proj(data["insulin"].to(self.device))
        sugar = self.sugar_proj(data["sugar"].to(self.device))
        drug = self.drug_proj(data["drug"].to(self.device))
        day = self.day_embedding(data["day"].to(self.device))
        time = self.time_embedding(data["time"].to(self.device))
        examination = self.patient2hidden(vec)
        _, seq_len, _ = sugar.shape
        examination = examination.unsqueeze(1).repeat((1, seq_len, 1))

        x = torch.cat((insulin, sugar, drug, day, time, examination), dim=2)
        x = torch.tanh(x)
        out, _ = self.encoder(x)
        out = torch.cat((out, x), dim=-1)
        out = self.out_proj(out)

        targets = data["target"]
        nonzero_pos = (targets > 0).float()
        nonzero_pos_cnt = torch.sum(nonzero_pos).item()
        if nonzero_pos_cnt == 0:
            return torch.tensor(0.0, requires_grad=True), None
        diff = nonzero_pos*abs(targets - out)
        loss = torch.sum(diff)/nonzero_pos_cnt
        return loss, out
