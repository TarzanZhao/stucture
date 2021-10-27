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
from train_FE import FEdataset, NaiveLinear, BOWcallback
from train_bow import NaiveLinear, BOWcallback


class FastTextdataset(FEdataset):
    def __init__(self, hash_dim=1000, mode="train", base_dir="yelp_small/"):
        super().__init__(Ngram=True, PunctuationC=False, UpperCase=False, mode=mode, base_dir="yelp_small/")
        # self.bows
        self.hash_dim = hash_dim
        self.hash_bows = []
        for cnts in self.bows:
            vec = [0 for _ in range(self.hash_dim)]
            for i in range(len(cnts)):
                vec[i%hash_dim] += cnts[i]
            self.hash_bows.append(vec)
        self.input_dim = hash_dim + len(self.Ngram[0])
        print("FastTextdataset dim=", self.input_dim)

    def __getitem__(self, index):
        X = self.hash_bows[index] + self.Ngram[index]
        return torch.FloatTensor(X), torch.tensor(self.labels[index]-1, dtype=torch.int64)

class FastTextModel(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 5)
        self.linear2 = nn.Linear(100, 5)
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()

    def get_loss(self, data):
        X, Y = data[0].to(self.device), data[1].to(self.device)
        out = self.linear(X)
        cnt = X.sum(-1)
        # print(cnt)
        # exit()
        out = out/cnt.view(-1,1)*80#*60 works
        # print(abs(out).sum(-1))
        # out = self.linear2(out)
        # exit()
        loss = self.criterion(out, Y)
        return loss, out.argmax(-1)

def model_provider(args):
    model = FastTextModel(args, args.input_dim)
    model = model.to(args.device)
    return model

def optimizer_provider(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def lr_scheduler_func(optimizer, args):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=0)
    return scheduler

def data_provider_func(args, model):
    train_dataset = FastTextdataset(1000,mode='train')
    args.input_dim = train_dataset.input_dim
    test_dataset = FastTextdataset(1000,mode='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def callback_func(args):
    return BOWcallback("train"), BOWcallback("valid")

def test_data_provider_func(args):
    test_dataset = FastTextdataset(1000,mode='test')
    args.input_dim = test_dataset.input_dim
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
