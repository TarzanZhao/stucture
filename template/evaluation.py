import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from template.arguments import get_args
from template.tools.logger import initialize_logger, get_logger

def preeval(model_provider_func, data_provider_func, callback_func):
    args = get_args()
    dataset, dataloader = data_provider_func(args)
    model = model_provider_func(args)
    model.load_state_dict(torch.load(args.save_folder+"/best.pth", map_location=model.device))
    callback_eval = callback_func(args)
    callback_eval.initialize(model=model)
    evaluate(model, dataloader, callback_eval)

@torch.no_grad()
def evaluate(model, dataloader, callback):
    model.eval()
    callback.start_epoch()
    with torch.no_grad():
        with tqdm(dataloader, desc="testing") as pbar:
            for idx, data in enumerate(pbar):
                callback.start_step(data=data)
                loss, output = model.get_loss(data)
                callback.end_step(loss=loss, output=output)
    callback.end_epoch()

if __name__ == "__main__":
    pass
