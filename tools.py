import logging
from torch.utils.tensorboard import SummaryWriter


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
def tools_get_tensorboard_writer():
    global tools_tensorboard_writer_judger
    if not tools_tensorboard_writer_judger:
        writer = SummaryWriter(log_dir='./logs')
    return writer