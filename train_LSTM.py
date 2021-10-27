#implement callback, model, optimizer, lr_scheduler, dataloader before going into this main process.
from template.models.lstm import RNN_v2
from template.train import pretrain
from template.evaluation import preeval
from template.initialize import initialize
from template.tools.logger import get_logger
from template.arguments import get_args
import torch
from torch.utils.data import DataLoader
from torch import optim
from template.modules.callback import mycallback
from template.datasets.dataset import BaseDataset
from nltk.corpus import stopwords
import json
import os
import torch.nn as nn
from template.modules.callback import callbackBase
import numpy as np
import copy
from train_CNN import CNNdataset, init_with_glove
from train_bow import BOWdataset, BOWcallback

class LSTMModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size+1, embedding_dim=100, padding_idx=args.vocab_size)
        self.lstm = nn.LSTM(input_size=100, hidden_size=args.hidden, num_layers=args.lstm_N, batch_first=True)
        self.final_linear = nn.Linear(args.hidden, 5)

        for m in self.parameters():
            if m.dim() > 1:
                nn.init.xavier_normal_(m)
        
        self.criterion = nn.CrossEntropyLoss()
        self.args = args

    def get_loss(self, data):
        X, Y = data[0].to(self.device), data[1].to(self.device)
        X_bed = self.word_embedding(X)#bsz*200*100
        hidden = self.zeroHidden(X_bed.shape[0])
        output, hidden = self.lstm(X_bed, hidden)
        # print(output.shape, hidden[0].shape, hidden[1].shape)
        tmp = []
        for i in range(X_bed.shape[0]):
            for j in range(199, -1, -1):
                if X[i,j] != self.args.vocab_size:
                    if self.args.take_output == "maxpooling":
                        tmp.append(output[i,:j+1, :].max(dim=0)[0])
                    elif self.args.take_output == "avepooling":
                        tmp.append(output[i,:j+1, :].mean(dim=0))
                    else:
                        tmp.append(output[i,j, :])
                    break
        tmp = torch.stack(tmp)#bsz*hidden
        # print(tmp.shape)
        out = self.final_linear(tmp)
        # print(out.shape)
        loss = self.criterion(out, Y)
        return loss, out.argmax(-1)

    def zeroHidden(self, bsz):
        a = self.args
        return (torch.zeros(a.lstm_N, bsz, a.hidden, device=a.device), torch.zeros(a.lstm_N, bsz, a.hidden, device=a.device))

def model_provider(args):
    model = LSTMModel(args)
    if args.noglove == 0:
        init_with_glove(args.vocab_id_map, model.word_embedding)
    model = model.to(args.device)
    return model

def optimizer_provider(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def lr_scheduler_func(optimizer, args):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=0)
    return scheduler

def data_provider_func(args, model):
    train_dataset = CNNdataset(mode='train')
    args.input_dim = train_dataset.input_dim
    args.vocab_size = train_dataset.vocab_size
    args.vocab_id_map = train_dataset.vocab_id_map
    test_dataset = CNNdataset(mode='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def callback_func(args):
    return BOWcallback("train"), BOWcallback("valid")

def test_data_provider_func(args):
    test_dataset = CNNdataset(mode='test')
    args.input_dim = test_dataset.input_dim
    args.vocab_size = test_dataset.vocab_size
    args.vocab_id_map = test_dataset.vocab_id_map
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return test_dataset, test_dataloader
def test_callback_func(args):
    return BOWcallback("test")

if __name__ == "__main__":
    initialize()
    args = get_args()
    if args.train == 1:
        pretrain(model_provider, optimizer_provider, lr_scheduler_func, data_provider_func, callback_func)
    else:
        preeval(model_provider, test_data_provider_func, test_callback_func)
