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


class IMBaseDataset(Dataset):
    def __init__(self, base_dir="IMdata/", device=torch.device("cpu"), mode="train", patient_ids_path="filtered_patients_id.json", patients_path="patients", split_factor=0.8, insulin_dim=3):
        super().__init__()
        self.base_dir = base_dir
        self.insulin_names = json.load(open(base_dir+"all_insulins.json", "r"))
        self.insulin_cnt = len(self.insulin_names)
        self.insulin_dim = insulin_dim
        self.drug_names = json.load(open(base_dir+"needed_drugs.json", "r"))
        self.drug_cnt = len(self.drug_names)
        self.examinations = json.load(
            open(base_dir+"required_examinations.json", "r"))["data"]
        self.examinations = ["age", "gender"] + list(self.examinations.keys())
        self.examinations_cnt = len(self.examinations)

        self.patient_ids = json.load(open(base_dir+patient_ids_path, "r"))
        split_pos = int(len(self.patient_ids)*split_factor)
        if mode == "train":
            self.patient_ids = self.patient_ids[:split_pos]
        else:
            self.patient_ids = self.patient_ids[split_pos:]
        self.patient_cnt = len(self.patient_ids)
        self.examination_dim = len(self.examinations)*2
        self.patients_path = patients_path
        self.data = []
        for patient_id in self.patient_ids:
            self.data.append(json.load(open(os.path.join(self.base_dir, f"{patients_path}/{patient_id}.json"), 'r')))

    def dict_to_drugdata(self, data, mode="start"):
        new_data = {"action": "drug_"+mode, "vec": torch.zeros((self.drug_cnt+1,))}
        new_data["day"] = torch.tensor(
            data["day"] if mode == "start" else data["day_end"], dtype=torch.long)
        new_data["time"] = torch.tensor(
            data["time"] if mode == "start" else data["day_time"], dtype=torch.float32)
        new_data["vec"][data["drug_id"]] = 1
        new_data["vec"][-1] = data["value"]
        return new_data

    def dict_to_insulindata(self, data):
        new_data = {"action": "insulin", "time": torch.tensor(data["time"], dtype=torch.float32),
                    "day": torch.tensor(data["day"], dtype=torch.long)}
        str_to_id = {"basal": 0, "premix": 1, "shot": 2}
        vec = torch.zeros((self.insulin_dim,))
        vec[str_to_id[data["type"]]] = data["value"]
        new_data["vec"] = vec
        predict_vec = torch.zeros((self.insulin_dim, ))
        predict_vec[str_to_id[data["type"]]] = 1
        new_data["predict_vec"] = predict_vec
        return new_data

    def dict_to_insulinsugardata(self, insulin_and_sugar, data):
        new_data = {"action": "insulinsugar", "day": torch.tensor(data["day"], dtype=torch.long),
                    "time": torch.tensor(data["time"], dtype=torch.float32), "insulin_and_sugar": insulin_and_sugar}
        return new_data

    def dict_to_sugardata(self, data, mode=0):
        new_data = {"action": "sugar", "time": torch.tensor(data["time"], dtype=torch.float32), "day": torch.tensor(data["day"], dtype=torch.long)}
        vec = torch.zeros((2,))
        vec[0] = data["value"]
        vec[-1] = mode
        new_data["vec"] = vec
        return new_data

    def get_examination_vec(self, data):
        vec = []
        for key in self.examinations:
            if not key in data.keys():
                vec.append(0)
                vec.append(1)
            else:
                vec.append(data[key])
                vec.append(0)
        return torch.tensor(vec, dtype=torch.float32)

    def time_diff(self, day1, time1, day2, time2):
        diff = (day2-day1)*24+time2-time1
        return diff

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return self.patient_cnt

class FullTensorDataset_v001(IMBaseDataset):
    def __init__(self, base_dir="IMdata/", device=torch.device("cpu"), mode="train", patient_ids_path="patients_id.json", patients_path="version001", split_factor=0.8, insulin_dim=3):
        super().__init__(base_dir, device, mode, patient_ids_path, patients_path, split_factor, insulin_dim)
        self.patients_path = patients_path
        self.insulin_to_type = json.load(open(base_dir+"insulin_to_type.json", "r"))
        self.classification = self.insulin_to_type["classifications"][0]
        self.insulin_to_type_id = dict()
        for key, value in self.insulin_to_type["insulin_to_type"].items():
            self.insulin_to_type_id[key] = self.classification.index(value[0])

    def __getitem__(self, index):
        p_id = self.patient_ids[index]
        p_data = json.load(open(self.base_dir+self.patients_path+f"/{p_id}.json", 'r'))
        examination_vector = self.get_examination_vec(p_data['basic_info'])

        days = p_data["days"]
        mx_day = max([int(key) for key in days.keys()])

        timeline_sugar = []
        timeline_insulin = []
        timeline_drug = []
        timeline_target = []
        timeline_day = []
        timeline_time = []
        for day in range(1, mx_day+1):
            seven_points = days[str(day)]["daily_routine"]
            for key, data in seven_points.items():
                if data["measuring_blood_sugar"] is None:
                    sugar_vec = [0,1]
                else:
                    sugar_vec = [data["measuring_blood_sugar"]["value"], 0]
                timeline_sugar.append(sugar_vec)

                insulin_vec = [0 for i in range(self.insulin_dim)]
                for insulin in data["injecting_insulin"]:
                    insulin_vec[self.insulin_to_type_id[insulin["insulin"]]] += insulin["value"]

                timeline_insulin.append(insulin_vec)
                
                drug_vec = [0 for i in range(28)]
                for drug in data["taking_hypoglycemic_drugs"]:
                    drug_vec[ self.drug_names.index(drug["drug"])] += drug["value"]
                timeline_drug.append(drug_vec)
                timeline_day.append(day)
                timeline_time.append(int(key))
                if len(timeline_sugar) > 1:
                    if sugar_vec[-1] == 1:
                        timeline_target.append([0])
                    else:
                        timeline_target.append([sugar_vec[0]])

        if len(timeline_sugar) > 0:
            timeline_target.append([0])

        timeline_sugar = torch.tensor(timeline_sugar)
        timeline_insulin = torch.tensor(timeline_insulin)
        timeline_drug = torch.tensor(timeline_drug)
        timeline_day = torch.tensor(timeline_day, dtype=torch.long)
        timeline_time = torch.tensor(timeline_time, dtype=torch.long)
        timeline_target = torch.tensor(timeline_target)
        return {
            "p_id": p_id,
            "examination_vec": examination_vector,
            "sugar_vec": timeline_sugar,
            "insulin_vec": timeline_insulin,
            "drug_vec": timeline_drug,
            "day_vec": timeline_day,
            "time_vec": timeline_time,
            "target_vec": timeline_target
        }

