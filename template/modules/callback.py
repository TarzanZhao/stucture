import numpy as np
from template.tools.logger import get_logger
import torch
from template.arguments import get_args
import os
import cv2
from torchvision.utils import save_image
import copy

class callbackBase:
    def __init__(self):
        self.args = get_args()
        pass

    def initialize(self, model, train_dataloader=None, test_dataloader=None, optimizer=None, lr_scheduler=None):
        self.model = model

    def start_epoch(self, **data):
        pass

    def end_epoch(self, **data):
        pass

    def start_step(self, **data):
        pass

    def end_step(self, **data):
        pass

    def finish(self, **data):
        pass

class callbackList(callbackBase):

    def __init__(self, callbacklist):
        super().__init__()
        self.callbacklist = callbacklist

    def start_epoch(self, **data):
        for callback in self.callbacklist:
            callback.start_epoch(**data)

    def end_epoch(self, **data):
        for callback in self.callbacklist:
            callback.end_epoch(**data)

    def start_step(self, **data):
        for callback in self.callbacklist:
            callback.start_test_epoch(**data)

    def end_step(self, **data):
        for callback in self.callbacklist:
            callback.end_test_epoch(**data)

class trackloss(callbackBase):
    def __init__(self):
        super().__init__()
        self.epoch_loss = []
        self.step_loss = []
        self.num_epochs = 0
        self.log = get_logger()

    def start_epoch(self, **data):
        self.step_loss = []
        self.num_epochs += 1

    def end_epoch(self, **data):
        loss = np.mean(self.step_loss)
        self.log.info(f"{self.num_epochs} {loss}\n")
        self.epoch_loss.append(loss)

    def end_step(self, **data):
        loss = data['loss']
        self.step_loss.append(loss.item())

class mycallback(callbackBase):
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

    def start_epoch(self, **data):
        self.step_loss = []
        self.num_epochs += 1
        self.num_steps = 0
        self.show_loss = 0

    def end_epoch(self, **data):
        loss = np.mean(self.step_loss)
        self.log.info( self.mode + f" {self.num_epochs} {loss}")
        if loss < self.best_loss:
            self.best_loss = loss
            if self.mode == "test":
                torch.save(self.model.state_dict(), os.path.join(self.args.save_folder,"best.pth") )
                self.log.info(f"save model.")
        self.epoch_loss.append(loss)

    def start_step(self, **data):
        self.data = copy.deepcopy(data['data'])
        self.num_steps += 1
        if self.num_steps % 500==0:
            torch.save(self.model.state_dict(), os.path.join(self.args.save_folder,"current.pth") )

    def end_step(self, **data):
        loss = data['loss']
        output = data['output']
        targets = self.data["target"][:,:-1,:]
        bsz, seq, _ = targets.shape

        if self.mode=="eval":
            self.log.info("------------------------")
            for i in range(bsz):
                self.log.info("-------------")
                for j in range(seq):
                    self.log.info(f"{targets[i,j,0]} {output[i,j,0]}")
            if self.num_steps == 5:
                exit()
        self.step_loss.append(loss.item())
        self.show_loss = (self.show_loss*(self.num_steps-1) + loss.item())/self.num_steps



