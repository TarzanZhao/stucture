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
from train_bow import BOWdataset, NaiveLinear, BOWcallback

class FEdataset(BOWdataset):
    def __init__(self, Ngram=True, PunctuationC=True, UpperCase=True, mode="train", base_dir="yelp_small/"):
        super().__init__(mode=mode, base_dir="yelp_small/")
        self.use_Ngram = Ngram
        if Ngram:
            self.Ngram_feature()

        self.use_PunctuationC = PunctuationC
        if PunctuationC:
            self.PunctuationC_feature()

        self.use_UpperCase = UpperCase
        if UpperCase:
            self.UpperCase_feature()

    def Ngram_feature(self):
        if os.path.exists(self.base_dir+self.mode+"_Ngram.json"):
            self.Ngram = json.load(open(self.base_dir+self.mode+"_Ngram.json", "r"))
        else:
            self.cnt = {}
            self.words = json.load(open(self.base_dir+self.mode+"_bow_words.json","r"))
            for index, words in enumerate(self.words):
                for i in range(len(words)-1):
                    wd_p = words[i]+"$"+words[i+1]
                    if wd_p not in self.cnt.keys():
                        self.cnt[wd_p] = [0 for _ in range(5)]
                    self.cnt[wd_p][self.labels[index]-1] += 1

            self.dict = {}
            if self.mode == "train":
                for wd_p, ct in self.cnt.items():
                    ct_sum = sum(ct)
                    mx = max(ct)
                    if ct_sum > 6 and mx/ct_sum>0.6:
                        self.dict[wd_p] = len(self.dict)
                json.dump(self.dict, open(self.base_dir+"Ngram_dict.json", "w"))
            else:
                self.dict = json.load(open(self.base_dir+"Ngram_dict.json", "r"))
            
            dict_sz = len(self.dict)
            self.Ngram = []
            for words in self.words:
                vec = [0 for _ in range(dict_sz)]
                for i in range(len(words)-1):
                    wd_p = words[i]+"$"+words[i+1]
                    if wd_p in self.dict.keys():
                        vec[self.dict[wd_p]] += 1
                self.Ngram.append(vec)
            json.dump(self.Ngram, open(self.base_dir+self.mode+"_Ngram.json", "w"))
        print("Ngram dim=", len(self.Ngram[0]))
        self.input_dim += len(self.Ngram[0])

    def PunctuationC_feature(self):
        if os.path.exists(self.base_dir+self.mode+"_PunctuationC.json"):
            self.PunctuationC = json.load(open(self.base_dir+self.mode+"_PunctuationC.json", "r"))
        else:
            puncs = ['?','!',',','.']
            self.PunctuationC = []
            for index, sentence in enumerate(self.sentences):
                vec = [0,0,0,0]
                for c in sentence:
                    for i in range(4):
                        if puncs[i] == c:
                            vec[i] += 1
                            break
                self.PunctuationC.append(vec)
            json.dump(self.PunctuationC, open(self.base_dir+self.mode+"_PunctuationC.json", "w"))
        print("PunctuationC dim=", len(self.PunctuationC[0]))
        self.input_dim += len(self.PunctuationC[0])    

    def UpperCase_feature(self):
        if os.path.exists(self.base_dir+self.mode+"_UpperCase.json"):
            self.UpperCase = json.load(open(self.base_dir+self.mode+"_UpperCase.json", "r"))
        else:
            self.UpperCase = []
            for index, sentence in enumerate(self.sentences):
                vec = [0]
                for c in sentence:
                    if 'A'<=c and c<='Z':
                        vec[0] += 1
                self.UpperCase.append(vec)
            json.dump(self.UpperCase, open(self.base_dir+self.mode+"_UpperCase.json", "w"))
        print("UpperCase dim=", len(self.UpperCase[0]))
        self.input_dim += len(self.UpperCase[0])

    def __getitem__(self, index):
        X = self.bows[index]
        if self.use_Ngram:
            X += self.Ngram[index]
        if self.use_PunctuationC:
            X += self.PunctuationC[index]
        if self.use_UpperCase:
            X += self.UpperCase[index]
        return torch.FloatTensor(X), torch.tensor(self.labels[index]-1, dtype=torch.int64)

def model_provider(args):
    model = NaiveLinear(args, args.input_dim)
    model = model.to(args.device)
    return model

def optimizer_provider(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def lr_scheduler_func(optimizer, args):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=0)
    return scheduler

def data_provider_func(args, model):
    train_dataset = FEdataset(args.Ngram, args.Punc, args.UpperCase, mode='train')
    args.input_dim = train_dataset.input_dim
    test_dataset = FEdataset(args.Ngram, args.Punc, args.UpperCase, mode='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def callback_func(args):
    return BOWcallback("train"), BOWcallback("valid")

def test_data_provider_func(args):
    test_dataset = FEdataset(args.Ngram, args.Punc, args.UpperCase, mode='test')
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
