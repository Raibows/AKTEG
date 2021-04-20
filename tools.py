import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


tools_global_logger_judger = None
def tools_get_logger(name:str='test'):
    global tools_global_logger_judger
    if not tools_global_logger_judger:
        fmt = "{asctime} {name:8s} {msg}"
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        bf = logging.Formatter(fmt, style='{', datefmt='%y-%m-%d %H:%M:%S')
        handler.setFormatter(bf)
        root.addHandler(handler)
    return logging.getLogger(name)

tools_tensorboard_writer_judger = None
def tools_get_tensorboard_writer(log_dir=None):
    if not log_dir:
        log_dir = f'./logs/{tools_get_time()}'
    global tools_tensorboard_writer_judger
    if not tools_tensorboard_writer_judger:
        writer = SummaryWriter(log_dir=log_dir)
    return writer


def tools_get_time():
    return datetime.now().strftime("%y-%m-%d-%H_%M_%S")

def tools_setup_seed(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    tools_get_logger('tools').info(f"set the seed to {seed}")