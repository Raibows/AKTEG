import logging
import re
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import shutil
import torch
import sys


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
    start_time = tools_get_time()
    if not log_dir:
        log_dir = f'./logs/{dir_pre}/{start_time}'
    if dir_pre not in tools_tensorboard_writers:
        tools_tensorboard_writers[dir_pre] = SummaryWriter(log_dir=log_dir)
    return tools_tensorboard_writers[dir_pre], log_dir, start_time


def tools_get_time():
    return datetime.now().strftime("%y-%m-%d-%H_%M_%S")

def tools_setup_seed(seed):
    import torch
    import numpy as np
    import random
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
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

def tools_copy_all_suffix_files(target_dir, source_dir='.', suffix='.py'):
    if target_dir[-1] != '/': target_dir += '/'
    tools_make_dir(target_dir)
    src_files = os.listdir(source_dir)
    for file in src_files:
        if file.endswith(suffix):
            tools_copy_file(f'{source_dir}/{file}', f'{target_dir}{file}')


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

def tools_parse_eval_file(path):
    """
    epoch 000 train_loss 7.2052 test_loss 7.0217 novelty 0.8735
    div1 0.0008 div2 0.0040 bleu2 0.0231 bleu3 0.0023 bleu4 0.0016
    mixbleu2 0.0615 mixbleu3 0.0182 mixbleu4 0.0093
    -------
    :param path:
    :return:
    """
    train_loss, _, novelty = [], None, []
    div1, div2, bleu2, _, _ = [], [], [], None, None
    _, _, mixbleu4 = None, None, []
    with open(path, 'r', encoding='utf-8') as file:
        t = 0
        pat = '\d{1}\.\d+'
        for i, line in enumerate(file):
            if (i+1) % 4 == 0:
                t = 0
                continue
            t += 1
            line = line.strip('\n').strip()
            res = re.findall(pat, line)
            res = list(map(lambda x: float(x), res))
            if t == 1:
                tl, _, no = res
                train_loss.append(tl)
                novelty.append(no)
            if t == 2:
                d1, d2, b2, _, _ = res
                div1.append(d1)
                div2.append(d2)
                bleu2.append(b2)
            if t == 3:
                _, _, m4 = res
                mixbleu4.append(m4)

    epoch = len(train_loss)
    return epoch, train_loss, novelty, div1, div2, bleu2, mixbleu4


def tools_check_if_in_debug_mode():
    gettrace = getattr(sys, 'gettrace', lambda: None)
    return gettrace() is not None

if __name__ == '__main__':
    # path = 'logs/pretrain_G_magic/21-05-05-17_41_33/epoch_65.predictions'
    # _, _, generated = tools_parse_log_file(path)
    # print(generated)
    path = 'logs/knowledge/21-05-22-15_01_32/evaluate.log'
    epoch, train_loss, novelty, div1, div2, bleu2, mixbleu4 = tools_parse_eval_file(path)
    print(train_loss)
    pass


