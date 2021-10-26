from template.arguments import parse_args, get_args, print_args
from template.tools.logger import initialize_logger, get_logger
import torch
import os
import numpy as np
import getpass


def post_process_args(args):
    if args.save_folder == None:
        args.save_folder = "expe/tmp"
    if args.device == -1:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.device}")
    return args


def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)
    os.removedirs(path)


def initialize():
    user_name = getpass.getuser()
    if user_name == 'hewanrong_chmed':
        torch.multiprocessing.set_start_method('forkserver', force=True)
    else:
        torch.multiprocessing.set_start_method('spawn')
    parse_args(post_process_args=post_process_args)
    args = get_args()

    if args.go_on_train == 0:
        if os.path.exists(args.save_folder):
            ans = input(
                "Path already exists, do you want to cover it?(0/1) : ")
            if ans != "1":
                print("Code Stops.")
                exit()
            else:
                del_file(args.save_folder)
        os.makedirs(args.save_folder)

    initialize_logger(args.save_folder)
    print_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
