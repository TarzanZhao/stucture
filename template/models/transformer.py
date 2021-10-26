import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import time
from torch.autograd import Variable
import torch.optim as optim
from template.arguments import get_args
import numpy as np
from template.models.modules import NaiveFE, TimeEmbedding

####################model structure#########################

class DecoderLayerWithoutEncoder(nn.Module):
    "DecoderWithoutEncoder is made of self-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayerWithoutEncoder, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderWithoutEncoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(DecoderWithoutEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class OnlyDecoderModel(nn.Module):
    """
    Only Decoder architecture.
    """
    def __init__(self, decoder, embed, generator):
        super(OnlyDecoderModel, self).__init__()
        self.decoder = decoder
        self.embed = embed
        self.generator = generator
        
    def forward(self, x, mask):
        "Take in and process masked sequences."
        return self.generator(self.decoder(self.embed(x), mask))

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab2):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab2)
        self.x = None
    def forward(self, x):
        self.x = x.clone().detach()
        return self.proj(x)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

########################attention mechanism########################

def attention(query, key, value, mask=None, dropout=None, d_k=100):
    "Compute 'Scaled Dot Product Attention'"
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.args = get_args()

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        nbatches = query.size(0) #self.nbatches
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, d_k=self.d_k)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)# batch * len * d_model

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x))))

########################make model########################

def make_transformer(N=8, d_model=200, d_ff=200, h=4, dropout=0.35):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = DecoderWithoutEncoder(DecoderLayerWithoutEncoder(d_model, c(attn), c(ff), dropout), N)
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Transformer_v1(nn.Module):
    def __init__(self):
        super(Transformer_v1, self).__init__()
        self.args = get_args()
        self.transformer = make_transformer(self.args.tf_N, self.args.tf_d_hidden, self.args.tf_d_ff, self.args.tf_n_h, self.args.tf_dropout)

        self.examination_FE = NaiveFE(1, self.args.examination_dim, self.args.tf_d_hidden)
        self.insulin_FE = NaiveFE(1, self.args.insulin_dim, self.args.tf_d_hidden)
        self.sugar_FE = NaiveFE(1, self.args.sugar_dim, self.args.tf_d_hidden)
        self.drug_FE = NaiveFE(1, self.args.drug_dim, self.args.tf_d_hidden)
        self.day_embedding = nn.Embedding(self.args.max_days, self.args.day_dim, padding_idx=-1)
        self.time_embedding = TimeEmbedding(self.args.time_dim)
        self.proj_before_transformer = NaiveFE(3,self.args.tf_d_hidden*4+self.args.day_dim*2+self.args.time_dim*2, 
                                          self.args.tf_d_hidden*4+self.args.day_dim*2+self.args.time_dim*2, 
                                          self.args.tf_d_hidden)
        self.proj_after_transformer = nn.Linear(self.args.tf_d_hidden, 1)
        
    def get_loss(self, data):
        examination_data = self.examination_FE(data["examination"])
        sugar_data = self.sugar_FE(data["sugar"])[:,:-1,:]
        insulin_data = self.insulin_FE(data["insulin"])[:,:-1,:]
        drug_data = self.drug_FE(data["drug"])[:,:-1,:]
        day_data = self.day_embedding(data["day"])[:,:-1,:]
        time_data = self.time_embedding(data["time"])[:,:-1,:]
        nx_day_data = self.day_embedding(data["day"][:,1:])
        nx_time_data = self.time_embedding(data["time"][:,1:])
        bsz, seq, _ = sugar_data.shape
        examination_data = examination_data.unsqueeze(1).repeat(1, seq, 1)
        # print(examination_data.shape, sugar_data.shape, insulin_data.shape, drug_data.shape, time_data.shape, nx_day_data.shape, nx_time_data.shape)

        input_data = torch.cat((examination_data, sugar_data, insulin_data, drug_data, day_data, time_data, nx_day_data, nx_time_data), dim=-1)
        input_data = F.leaky_relu(self.proj_before_transformer(input_data))

        bsz, seq, _ = input_data.shape
        mask = subsequent_mask(seq)

        output_data = self.transformer(input_data, mask)
        output_data = self.proj_after_transformer(output_data)

        targets = data["target"][:,:-1,:]
        nonzero_pos = (targets > 0).float()
        nonzero_pos_cnt = torch.sum(nonzero_pos).item()
        if nonzero_pos_cnt == 0:
            return torch.tensor(0.0, requires_grad=True), None
        diff = nonzero_pos*abs(targets - output_data)
        loss = torch.sum(diff)/nonzero_pos_cnt
        return loss, output_data


class Transformer_v2(nn.Module):
    def __init__(self):
        super(Transformer_v2, self).__init__()
        self.args = get_args()
        self.transformer = make_transformer(self.args.tf_N, self.args.tf_d_hidden, self.args.tf_d_ff, self.args.tf_n_h, self.args.tf_dropout)

        self.examination_FE = NaiveFE(1, self.args.examination_dim, self.args.tf_d_hidden)
        self.insulin_FE = NaiveFE(1, self.args.insulin_dim, self.args.tf_d_hidden)
        self.sugar_FE = NaiveFE(1, self.args.sugar_dim, self.args.tf_d_hidden)
        self.drug_FE = NaiveFE(1, self.args.drug_dim, self.args.tf_d_hidden)
        self.day_embedding = nn.Embedding(self.args.max_days, self.args.day_dim, padding_idx=0)
        self.time_embedding = nn.Embedding(8, self.args.time_dim, padding_idx=-1)
        self.proj_before_transformer = NaiveFE(3,self.args.tf_d_hidden*4+self.args.day_dim+self.args.time_dim, 
                                          self.args.tf_d_hidden*4+self.args.day_dim+self.args.time_dim, 
                                          self.args.tf_d_hidden)
        self.proj_after_transformer = nn.Linear(self.args.tf_d_hidden, 1)
        
    def get_loss(self, data):
        examination_data = self.examination_FE(data["examination"])
        sugar_data = self.sugar_FE(data["sugar"])
        insulin_data = self.insulin_FE(data["insulin"])
        drug_data = self.drug_FE(data["drug"])
        day_data = self.day_embedding(data["day"])
        time_data = self.time_embedding(data["time"])

        bsz, seq, _ = sugar_data.shape
        examination_data = examination_data.unsqueeze(1).repeat(1, seq, 1)
        # print(examination_data.shape, sugar_data.shape, insulin_data.shape, drug_data.shape, time_data.shape, nx_day_data.shape, nx_time_data.shape)

        input_data = torch.cat((examination_data, sugar_data, insulin_data, drug_data, day_data, time_data), dim=-1)
        input_data = F.leaky_relu(self.proj_before_transformer(input_data))

        bsz, seq, _ = input_data.shape
        mask = subsequent_mask(seq)
        mask = mask.to(sugar_data.device)

        output_data = self.transformer(input_data, mask)
        output_data = self.proj_after_transformer(output_data)

        targets = data["target"]
        nonzero_pos = (targets > 0).float()
        nonzero_pos_cnt = torch.sum(nonzero_pos).item()
        if nonzero_pos_cnt == 0:
            return torch.tensor(0.0, requires_grad=True), None
        diff = nonzero_pos*abs(targets - output_data)
        loss = torch.sum(diff)/nonzero_pos_cnt
        return loss, output_data