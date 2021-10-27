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

class BOWdataset(BaseDataset):
    def __init__(self, mode="train", base_dir="yelp_small/"):
        super().__init__(mode=mode, base_dir="yelp_small/")
        if os.path.exists(base_dir+mode+"_bow.json"):
            self.bows = json.load(open(base_dir+mode+"_bow.json", "r"))
            self.input_dim = len(self.bows[0])
        else:
            self.words = []
            self.cnt = {}
            for sentence in self.sentences:
                self.words.append( self.preprocess(sentence) )
                for wd in self.words[-1]:
                    if wd not in self.cnt.keys():
                        self.cnt[wd] = 1 #len(self.dict)
                    else:
                        self.cnt[wd] += 1
            self.dict = {}
            json.dump(self.words, open(base_dir+mode+"_bow_words.json", "w"))
            if mode == "train":
                for wd, ct in self.cnt.items():
                    if ct > 10:
                        self.dict[wd] = len(self.dict)+1
                json.dump(self.dict, open(base_dir+"bow_dict.json", "w"))
            else:
                self.dict = json.load(open(base_dir+"bow_dict.json", "r"))
            
            dict_sz = len(self.dict)
            self.input_dim = dict_sz+1
            self.bows = []
            for sentence_words in self.words:
                vec = [0 for _ in range(dict_sz+1)]
                for wd in sentence_words:
                    if wd in self.dict.keys():
                        vec[self.dict[wd]] += 1
                    else:
                        vec[0] += 1
                self.bows.append(vec)
            json.dump(self.bows, open(base_dir+mode+"_bow.json", "w"))
        print("bows dim=", len(self.bows[0]))

    def __getitem__(self, index):
        return torch.FloatTensor(self.bows[index]), torch.tensor(self.labels[index]-1, dtype=torch.int64)

    def __len__(self):
        return len(self.bows)

    def preprocess(self, s):
        s = s.lower().strip()

        punctuations = [
                r'\'',
                r'\"',
                r'.',
                r',',
                r'(',
                r')',
                r'!',
                r'\?',
                r';',
                r':',
                '<',
                '>',
                '*',
                '&',
                '%',
                '-',
                '@','#','$','[',']','{','}','|','~']

        n_s = " "
        for c in s:
            if c>="a" and c<="z":
                n_s += c
            elif c in punctuations:
                continue
            elif n_s[-1] != " ":
                n_s += " "
        n_s = n_s.strip(" ").split(" ")
        sl = []
        stopw = set(stopwords.words('english'))
        for wd in n_s:
            if wd not in stopw and not any(c.isdigit() for c in wd):
                sl.append(wd)
        # print(sl)
        return sl

class NaiveLinear(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 5)
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()

    def get_loss(self, data):
        X, Y = data[0].to(self.device), data[1].to(self.device)
        # print(X.shape, Y.shape)
        out = self.linear(X)
        loss = self.criterion(out, Y)
        return loss, out.argmax(-1)

class BOWcallback(callbackBase):
    def __init__(self, mode):
        super().__init__()
        self.epoch_loss = []
        self.step_loss = []
        self.num_epochs = 0
        self.log = get_logger()
        self.mode = mode
        self.best_loss = 1e9
        self.num_steps = 0
        self.show_loss = 0
        self.hit = 0
        self.total = 0

    def start_epoch(self, **data):
        self.step_loss = []
        self.num_epochs += 1
        self.num_steps = 0
        self.show_loss = 0
        self.hit = 0
        self.total = 0

    def end_epoch(self, **data):
        loss = np.mean(self.step_loss)
        self.log.info( self.mode + f" {self.num_epochs} {loss} {self.hit/self.total}")
        if loss < self.best_loss:
            self.best_loss = loss
            if self.mode == "valid" or self.mode == "test":
                torch.save(self.model.state_dict(), os.path.join(self.args.save_folder,"best.pth") )
                self.log.info(f"save model.")
        self.epoch_loss.append(loss)

    def start_step(self, **data):
        self.data = copy.deepcopy(data['data'])
        self.num_steps += 1
        # if self.num_steps % 500==0:
        #     torch.save(self.model.state_dict(), os.path.join(self.args.save_folder,"current.pth") )

    def end_step(self, **data):
        loss = data['loss']
        self.step_loss.append(loss.item())
        self.show_loss = (self.show_loss*(self.num_steps-1) + loss.item())/self.num_steps
        pd = data["output"].cpu()
        y = self.data[1]
        # print(pd)
        # print(y)
        # print("")
        self.hit += (pd == y).sum().item()
        self.total += y.shape[0]

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
    train_dataset = BOWdataset(mode='train')
    args.input_dim = train_dataset.input_dim
    test_dataset = BOWdataset(mode='valid')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def callback_func(args):
    return BOWcallback("train"), BOWcallback("valid")

def test_data_provider_func(args):
    test_dataset = BOWdataset(mode='test')
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
