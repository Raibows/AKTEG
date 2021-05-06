import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import shutil
import torch



tools_loggers_set = {}
def tools_get_logger(name:str='test'):
    global tools_loggers_set
    if name not in tools_loggers_set:
        fmt = "{asctime} [{name}] {msg}"
        root = logging.getLogger(name=name)
        root.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        bf = logging.Formatter(fmt, style='{', datefmt='%y-%m-%d %H:%M:%S')
        handler.setFormatter(bf)
        root.addHandler(handler)
        tools_loggers_set[name] = root
    return tools_loggers_set[name]

tools_tensorboard_writers = {}
def tools_get_tensorboard_writer(log_dir=None, dir_pre='public'):
    global tools_tensorboard_writer
    if not log_dir:
        log_dir = f'./logs/{dir_pre}/{tools_get_time()}'
    if dir_pre not in tools_tensorboard_writers:
        tools_tensorboard_writers[dir_pre] = SummaryWriter(log_dir=log_dir)
    return tools_tensorboard_writers[dir_pre], log_dir


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

def tools_make_dir(path):
    t = None
    for i in range(len(path)-1, -1, -1):
        if path[i] == '/':
            t = i
            break
    if t:
        os.makedirs(path[:t], exist_ok=True)

def tools_copy_file(source_path, target_path):
    shutil.copy(source_path, target_path)

def tools_to_gpu(*params, device=torch.device('cpu')):
    return [p.to(device) for p in params]

def tools_save_pickle_obj(obj, path):
    import pickle
    tools_get_logger('tools').info(f"saving obj_type {type(obj)} to {path}")
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def tools_load_pickle_obj(path):
    import pickle
    tools_get_logger('tools').info(f"loading obj from {path}")
    with open(path, 'rb') as file:
        return pickle.load(file)

def tools_batch_idx2words(idxs, idx2word:dict):
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.tolist()
    if isinstance(idxs[0], list):
        temp = [' '.join([idx2word[i] for i in x]) for x in idxs]
    else:
        temp = ' '.join([idx2word[i] for i in idxs])
    return temp

def tools_write_log_to_file(fmt, value, path):
    tools_make_dir(path)
    with open(path, 'a', encoding='utf-8') as file:
        file.write(fmt.format(*value))
        file.write('\n')
        file.write('-*-'*25)
        file.write('\n')

def tools_parse_log_file(path):
    res = {0: [], 1: [], 2: []}
    t = 0
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if (i+1) % 4 == 0:
                t = 0
                continue
            line = line.strip('\n').strip()
            if line.startswith('epoch'): break
            res[t].append(line)
            t += 1
    # topic, target, generated
    return res[0], res[1], res[2]

if __name__ == '__main__':
    path = 'logs/pretrain_G_magic/21-05-05-17_41_33/epoch_65.predictions'
    _, _, generated = tools_parse_log_file(path)
    print(generated)



