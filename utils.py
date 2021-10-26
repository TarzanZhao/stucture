import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

def create_logger(save_path='', file_type='', level='debug', console=True):
    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    if console:
        cs = logging.StreamHandler()
        cs.setLevel(_level)
        logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def plot_single_curve(path, name, point, freq=1, xlabel='Epoch'):
    
    x = (np.arange(len(point)) + 1) * freq
    plt.plot(x, point, color='purple')
    plt.xlabel(xlabel)
    plt.ylabel(name)
    plt.savefig(os.path.join(path, name + '.png'))
    plt.close()

def plot_curve(path, name, train_point, val_point=None, val_freq=None):
    
    x_t = np.arange(len(train_point)) + 1
    if val_point is not None:
        x_v = (np.arange(len(val_point)) + 1) * val_freq
    
    plt.plot(x_t, train_point, color='blue', label='train')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if val_point is not None:
        plt.plot(x_v, val_point, color='red', label='valid')
    plt.legend()
    plt.savefig(os.path.join(path, name + '.png'))
    plt.close()

def plot_time_curve(path, name, pre_point, gt_point, mask=None):
    
    x_t = np.arange(len(pre_point))  
    if mask is not None:
        plt.plot(x_t[mask], gt_point[mask], color='red', label='ground truth')
        plt.plot(x_t[mask], pre_point[mask], color='blue', label='prediction')
    else:
        plt.plot(x_t, gt_point, color='red', label='ground truth')
        plt.plot(x_t, pre_point, color='blue', label='prediction')
    plt.xlabel('pos')
    plt.ylabel(name)
        
    plt.legend()
    plt.savefig(os.path.join(path, name + '.png'))
    plt.close()

def fetch_results():
    folders = [f"lstm_fake00{i}" for i in range(1,10)]+[f"lstm_fake0{i}" for i in range(10,12)] +\
              [f"lstm_noins_fake00{i}" for i in range(1,10)]+[f"lstm_noins_fake0{i}" for i in range(10,12)] +\
              [f"transformer_fake00{i}" for i in range(1,10)]+[f"transformer_fake0{i}" for i in range(10,12)] +\
              [f"transformer_noins_fake00{i}" for i in range(1,10)]+[f"transformer_noins_fake0{i}" for i in range(10,12)]

    outf = open("result_for_fake.txt","w")
    for folder in folders:
        path = "expe/"+folder+"/log.txt"
        print(path)
        with open(path, "r") as f:
            data = f.readlines()
            # print(data)
            result = "None"
            for idx,line in enumerate(data):
                if line == "save model.\n":
                    result = data[idx-1].strip("\n").split(" ")[-1]
            outf.write(folder+" : "+result+"\n")

# fetch_results()
