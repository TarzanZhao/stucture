#implement callback, model, optimizer, lr_scheduler, dataloader before going into this main process.
from template.models.lstm import RNN_v2
from template.train import pretrain
from template.initialize import initialize
from template.tools.logger import get_logger
from template.arguments import get_args
import torch
from torch.utils.data import DataLoader
from template.datasets.dataset import FullTensorDataset_v001
from torch import optim
from template.collate_fns import fulltensor_collate_fn
from template.modules.callback import mycallback

def model_provider(args):
    model = RNN_v2(device=args.device, examination_size=args.examination_dim, 
                examination_proj_size=args.examination_proj_dim,
                insulin_size=args.insulin_dim, insulin_proj_size=args.insulin_proj_dim,
                sugar_size=args.sugar_dim, sugar_proj_size=args.sugar_proj_dim,
                day_proj_size=args.day_dim,
                time_proj_size=args.time_dim,
                drug_size=args.drug_dim, drug_proj_size=args.drug_proj_dim,
                hidden_size=args.lstm_d_hidden, num_layers=args.lstm_N, output_size=1)
    model = model.to(args.device)
    return model

def optimizer_provider(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def lr_scheduler_func(optimizer, args):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    return scheduler

def data_provider_func(args, model):
    train_dataset = FullTensorDataset_v001("IMdata/", args.device, mode='train')
    test_dataset = FullTensorDataset_v001("IMdata/", args.device, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=fulltensor_collate_fn)#model.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=fulltensor_collate_fn)#model.collate_fn)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

def callback_func(args):
    return mycallback("train"), mycallback("test")

if __name__ == "__main__":
    initialize()
    pretrain(model_provider, optimizer_provider, lr_scheduler_func, data_provider_func, callback_func)
