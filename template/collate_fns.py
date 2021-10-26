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
from torchvision import transforms as T
from torch import nn


def fulltensor_collate_fn(samples):
    lens = [sample["sugar_vec"].shape[0] for sample in samples]
    mx_len = max(lens)+1

    examination_data = []
    sugar_data = []
    insulin_data = []
    drug_data = []
    day_data = []
    time_data = []
    target_data = []
    p_ids = []
    for sample, l in zip(samples, lens):
        p_ids.append(sample["p_id"])
        examination_data.append(sample["examination_vec"])
        # print(sample["examination_vec"].shape)
        sugar_data.append(torch.cat((sample["sugar_vec"], torch.zeros((mx_len-l, 2))), dim=0))
        # print(sample["insulin_vec"].shape)
        insulin_data.append(torch.cat((sample["insulin_vec"], torch.zeros((mx_len-l, 3))), dim=0))
        drug_data.append(torch.cat((sample["drug_vec"], torch.zeros((mx_len-l, 28))), dim=0))
        target_data.append(torch.cat((sample["target_vec"], torch.zeros((mx_len-l, 1))), dim=0))
        day_data.append(torch.cat((sample["day_vec"], torch.zeros((mx_len-l,), dtype=torch.long)), dim=0))
        time_data.append(torch.cat((sample["time_vec"], torch.zeros((mx_len-l,), dtype=torch.long)), dim=0))

    examination_data = torch.stack(examination_data)
    sugar_data = torch.stack(sugar_data)
    insulin_data = torch.stack(insulin_data)
    drug_data = torch.stack(drug_data)
    target_data = torch.stack(target_data)
    day_data = torch.stack(day_data)
    time_data = torch.stack(time_data)
    p_ids = torch.tensor(p_ids)

    return {
        "p_ids": p_ids,
        "examination": examination_data,
        "sugar": sugar_data,
        "insulin": insulin_data,
        "drug": drug_data,
        "day": day_data,
        "time": time_data,
        "target": target_data
    }