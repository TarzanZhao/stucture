import torch
from torch import tensor
import torch.nn as nn
from template.arguments import get_args
import torch.nn.functional as F
import math


class NaiveFE(nn.Module):
    def __init__(self, N, in_d, hidden, out_d=None):
        super(NaiveFE, self).__init__()
        assert N>=1
        if out_d is None:
            out_d = hidden

        self.N = N
        self.in_d = in_d
        self.hidden = hidden
        self.out_d  = out_d
        if N == 1:
            self.linears = nn.ModuleList([nn.Linear(in_d, out_d)])
        else:
            self.linears = nn.ModuleList( [nn.Linear(in_d, hidden)] + 
                      [nn.Linear(hidden, hidden) for _ in range(N-2)]+[nn.Linear(hidden, out_d)] )

    def forward(self, x):
        for i in range(self.N):
            x = self.linears[i](x)
            if i!=self.N-1:
                x = F.leaky_relu(x)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super(TimeEmbedding, self).__init__()
        assert time_dim%2==1
        self.time_dim = time_dim
        self.time_FE = NaiveFE(2, time_dim, time_dim)

    def sinusoidalPositionEmbedding(self, time_dim, t):
        pe = torch.zeros(t.shape+(time_dim,), device=t.device)
        t = t.unsqueeze(-1)
        div_term = torch.exp((torch.arange(0, time_dim, 2, dtype=torch.float, device=t.device) * -(math.log(10000.0) / time_dim)))
        # print(pe.shape, t.shape, div_term.shape, (t.float() * div_term).shape)
        pe[..., 0::2] = torch.sin(t.float() * div_term)
        pe[..., 1::2] = torch.cos(t.float() * div_term)
        return pe

    def forward(self, t):
        t = torch.cat( (self.sinusoidalPositionEmbedding(self.time_dim-1, t), t.unsqueeze(-1)), dim = -1)
        t = self.time_FE(t)
        return t