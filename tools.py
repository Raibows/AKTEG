import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



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

tools_tensorboard_writer_judger = None
def tools_get_tensorboard_writer(log_dir=None):
    global tools_tensorboard_writer_judger
    if not log_dir:
        log_dir = f'./logs/{tools_get_time()}'
    if not tools_tensorboard_writer_judger:
        tools_tensorboard_writer_judger = True
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

def tools_read_line_like_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.readlines()

def tools_write_line_like_file(path, datas):
    with open(path, 'w', encoding='utf-8') as file:
        file.writelines(datas)

def tools_split_line_like_datas(datas, split_ratio):
    import random
    tools_setup_seed(667)
    size = len(datas)
    split_num = int(size * split_ratio)
    temp = [i for i in range(size)]
    split_set = random.sample(temp, k=split_num)
    split_datas = [datas[i] for i in split_set]
    reserved = set(temp) - set(split_set)
    reserved_datas = [datas[i] for i in reserved]
    print(len(split_datas), len(reserved_datas))
    return reserved_datas, split_datas

def tools_split_train_test_set():
    from data import ZHIHU_dataset
    from config import config_zhihu_dataset as c
    all_dataset = ZHIHU_dataset(c.raw_data_path, c.topic_num_limit, c.essay_vocab_size, c.topic_threshold,
                                c.topic_padding_num, c.essay_padding_len, raw_mode=True)
    delete_indexs = all_dataset.limit_datas()
    datas = tools_read_line_like_file(c.raw_data_path)
    all = set([i for i in range(len(datas))]) - set(delete_indexs)
    datas = [datas[i] for i in all]
    train_datas, test_datas = tools_split_line_like_datas(datas, c.test_data_split_ratio)
    tools_write_line_like_file(c.train_data_path, train_datas)
    tools_write_line_like_file(c.test_data_path, test_datas)
    tools_get_logger('tools').info(f'train dataset num {len(train_datas)} test dataset num {len(test_datas)} '
                                   f'test / train is {len(test_datas)/len(train_datas):.4f} '
                                   f'test / (train+test) is {len(test_datas)/len(all):.4f}')

def tools_k_fold_split(all_dataset, batch_size, k=5):
    import random
    import torch
    from torch.utils.data import DataLoader
    all_size = len(all_dataset)
    fold_size = [all_size // k] * (k-1) + [all_size - (all_size // k) * (k-1)]
    all_set = set([i for i in range(all_size)])
    for i, fs in enumerate(fold_size):
        temp = set(random.sample(all_set, k=fs))
        all_set -= temp
        fold_size[i] = temp

    all_set = set([i for i in range(all_size)])
    kfolds = []
    for fs in fold_size:
        test = torch.utils.data.dataset.Subset(all_dataset, list(fs))
        train = torch.utils.data.dataset.Subset(all_dataset, list(all_set-fs))
        kfolds.append((
            DataLoader(train, batch_size=batch_size),
            DataLoader(test, batch_size=batch_size)
        ))
    return kfolds




if __name__ == '__main__':
    tools_split_train_test_set()