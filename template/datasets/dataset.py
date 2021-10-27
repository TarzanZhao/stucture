import json
import os
import copy
from numpy.lib.type_check import imag
from torch.utils.data import Dataset
import torch
import sys
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
import random
from torch import nn
import pandas as pd


class BaseDataset(Dataset):
    def __init__(self, mode="train", base_dir="yelp_small/"):
        super().__init__()
        self.dir = os.path.join(base_dir, mode+".csv")
        df = pd.read_csv(self.dir)
        col_names = df.columns.values.tolist()
        self.sentences = [col_names[1]]
        self.labels = [int(col_names[0])]
        for index, row in df.iterrows():
            self.labels.append(int(row[col_names[0]]))
            self.sentences.append(row[col_names[1]])
        self.base_dir = base_dir
        self.mode = mode
        print(mode+" dataset size=", len(self.sentences))

        vocabs = open("yelp_small/vocab.txt", "r").readlines()
        self.vocab_size = len(vocabs)
        self.vocab_id_map = {}
        for i, vb in enumerate(vocabs):
            wd = vb.strip()
            self.vocab_id_map[wd] = i
            # print(i, wd)
        # exit()


if __name__ == "__main__":
    dataset = BaseDataset()
    pass
        
