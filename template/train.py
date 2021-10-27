import torch
from template.arguments import get_args
from tqdm import tqdm
from template.evaluation import evaluate
from template.tools.logger import get_logger

def pretrain(model_provider_func, optimizer_provider_func, lr_scheduler_func=None, data_provide_func=None, callback_provide_func=None):

    args = get_args()
    log = get_logger()

    data_provider = data_provide_func(args, None)
    model = model_provider_func(args)
    log.info(str(model))
    optimizer = optimizer_provider_func(model, args)
    lr_scheduler = lr_scheduler_func(optimizer, args) if lr_scheduler_func is not None else None
    callback_train, callback_eval = callback_provide_func(args)

    train(model, data_provider, optimizer, lr_scheduler, callback_train, callback_eval)

def train_one_step(optimizer_step, model, data, optimizer):
    loss, output = model.get_loss(data)
    loss.backward()
    if optimizer_step == True:
        optimizer.step()
        optimizer.zero_grad()
    return loss, output

def train_one_epoch(epoch_idx, model, dataloader, optimizer, lr_scheduler, callback, args):
    """Train the model function."""
    args = get_args()
    model.train()

    callback.start_epoch(epoch_idx=epoch_idx)
    with tqdm(dataloader, desc="training") as pbar:
        for idx, data in enumerate(pbar):
            callback.start_step(data=data)
            loss, output= train_one_step( (idx+1) % args.optimizer_step_interval == 0 or (idx+1)==len(dataloader),  model, data, optimizer)
            callback.end_step(loss=loss, output=output)
            pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch_idx + 1, callback.show_loss, optimizer.param_groups[0]['lr']))
    if lr_scheduler is not None:
        lr_scheduler.step()
    callback.end_epoch()

def train(model, data_provider, optimizer, lr_scheduler, callback_train, callback_eval):
    args = get_args()
    train_dataset, train_dataloader, test_dataset, test_dataloader = data_provider
    callback_train.initialize(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, lr_scheduler=lr_scheduler)
    callback_eval.initialize(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, lr_scheduler=lr_scheduler)
    for idx in range(args.train_epochs):
        train_one_epoch(idx, model, train_dataloader, optimizer, lr_scheduler, callback_train, args)
        if (idx+1) % args.eval_interval == 0:
            evaluate(model, test_dataloader, callback_eval)
