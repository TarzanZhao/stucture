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
import torch.nn.functional as F
from template.modules.callback import callbackBase
import numpy as np
import copy
from train_bow import BOWdataset, NaiveLinear, BOWcallback

class CNNdataset(BOWdataset):
    def __init__(self, mode="train", base_dir="yelp_small/"):
        super().__init__(mode=mode, base_dir="yelp_small/")
        self.CNN_words = []
        for raw_sentence in self.sentences:
            sentence = self.preprocess(raw_sentence)
            # print(len(sentence), sentence)
            vec = [0 for _ in range(min(200, len(sentence)))]+[self.vocab_size for _ in range(min(200, len(sentence)), 200)]
            for i in range(min(200, len(sentence))):
                wd = sentence[i]
                if wd in self.vocab_id_map.keys():
                    vec[i] = self.vocab_id_map[wd]
            # print(vec)
            self.CNN_words.append(vec)
        # exit()

    def __getitem__(self, index):
        return torch.LongTensor(self.CNN_words[index]), torch.tensor(self.labels[index]-1, dtype=torch.int64)

    def preprocess(self, s):
        s = s.lower().strip()
        return s.split(" ")

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
            elif n_s[-1] != " ":
                n_s += " "
        n_s = n_s.strip(" ").split(" ")
        return n_s

class CNNModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=args.vocab_size+1, embedding_dim=100, padding_idx=args.vocab_size)
        self.cnn3 = nn.Conv2d(1, args.hidden, (3, 100))
        self.cnn5 = nn.Conv2d(1, args.hidden, (5, 100))
        self.cnn7 = nn.Conv2d(1, args.hidden, (7, 100))
        self.final_linear = nn.Linear(args.hidden*3, 5)
        self.device = args.device

        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        if args.activation == "tanh":
            self.activation = F.tanh
        elif args.activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.sigmoid

    def get_loss(self, data):
        X, Y = data[0].to(self.device), data[1].to(self.device)#bsz*200*100
        # print(X.shape)
        X = self.word_embedding(X)
        X = X.unsqueeze(1)#bsz*1*200*100
        # print(X.shape)
        X_3 = self.cnn3(X)#bsz*100*198*1
        X_5 = self.cnn5(X)#bsz*100*196*1 #???does activation matter?
        X_7 = self.cnn7(X)#bsz*100*194*1
        # print(X_3.shape, X_5.shape, X_7.shape)
        X_3 = self.activation(X_3.squeeze(-1))#bsz*100*198
        X_5 = self.activation(X_5.squeeze(-1))#bsz*100*196 #???does activation matter?
        X_7 = self.activation(X_7.squeeze(-1))#bsz*100*194
        # relu activation have better generalization than tanh
        # print(X_3.shape, X_5.shape, X_7.shape)

        if self.args.take_output == "maxpooling":
            X_3 = X_3.max(-1)[0] #bsz*100
            X_5 = X_5.max(-1)[0] #bsz*100
            X_7 = X_7.max(-1)[0] #bsz*100
        elif self.args.take_output == "avepooling":
            X_3 = X_3.mean(-1) #bsz*100
            X_5 = X_5.mean(-1) #bsz*100
            X_7 = X_7.mean(-1) #bsz*100
        else:
            X_3 = X_3[:,:, 0] #bsz*100
            X_5 = X_5[:,:, 0] #bsz*100
            X_7 = X_7[:,:, 0] #bsz*100

        # print(X_3.shape, X_5.shape, X_7.shape)

        X = torch.cat((X_3, X_5, X_7), dim = -1) #bsz*300
        # print(X.shape)
        # print(out)
        out = self.final_linear(X) #*200
        # print(out)
        # print(out, Y)
        # exit()
        loss = self.criterion(out, Y)
        return loss, out.argmax(-1)

def init_with_glove(vocab_id_map, word_embedding):
    glove_path = f"glove/glove.6B.100d.txt"
    data = open(glove_path, 'r').readlines()
    for line in data:
        tmp = line.strip().split(" ")
        word = tmp[0]
        if word not in vocab_id_map.keys():
            continue
        word_id = vocab_id_map[word]
        embed = []
        for t in tmp[1:]:
            embed.append(float(t))
        word_embedding.weight.data[word_id] = torch.FloatTensor(embed)
    word_embedding.weight.data = torch.Tensor(word_embedding.weight.data.numpy().tolist())

def model_provider(args):
    model = CNNModel(args)
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
