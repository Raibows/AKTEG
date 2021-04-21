import random


def read_line_like_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_line_like_file(path, datas):
    with open(path, 'w', encoding='utf-8') as file:
        file.writelines(datas)

def split_line_like_datas(datas, split_ratio):
    import random
    from tools import tools_setup_seed
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

def split_train_test_set():
    from data import ZHIHU_dataset
    from config import config_zhihu_dataset as c
    from tools import tools_get_logger
    all_dataset = ZHIHU_dataset(c.raw_data_path, c.topic_num_limit, c.essay_vocab_size, c.topic_threshold,
                                c.topic_padding_num, c.essay_padding_len)
    delete_indexs = all_dataset.limit_datas()
    datas = read_line_like_file(c.raw_data_path)
    all = set([i for i in range(len(datas))]) - set(delete_indexs)
    datas = [datas[i] for i in all]
    train_datas, test_datas = split_line_like_datas(datas, c.test_data_split_ratio)
    write_line_like_file(c.train_data_path, train_datas)
    write_line_like_file(c.test_data_path, test_datas)
    tools_get_logger('preprocess').info(f'train dataset num {len(train_datas)} test dataset num {len(test_datas)} '
                                   f'test / train is {len(test_datas)/len(train_datas):.4f} '
                                   f'test / (train+test) is {len(test_datas)/len(all):.4f}')

def k_fold_split(all_dataset, batch_size, k=5):
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


def process_word_dict_and_pretrained_wv(word_dict:dict, pretrained_wv: dict, wv_dim: int):
    from tools import tools_setup_seed
    tools_setup_seed(667)
    wv = [[0.0 for _ in range(wv_dim)] for i in word_dict]
    unk_num = 0
    for k, v in word_dict.items():
        if k in pretrained_wv: wv[v] = pretrained_wv[k]
        else:
            unk_num += 1
            wv[v] = [random.normalvariate(0, 0.3) for _ in range(wv_dim)]
    print(f'unk_num {unk_num} finding {len(word_dict)-unk_num} all {len(word_dict)}')
    return wv


def read_pretrained_word_vectors(path):
    pretrained_wv = {}
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip('\n\r').split(' ')
            wv = list(map(lambda x: float(x), line[1:-1]))
            pretrained_wv[line[0]] = wv
    return pretrained_wv

def save_pickle_obj(obj, path):
    import pickle
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle_obj(path):
    import pickle
    with open(path, 'rb') as file:
        return pickle.load(file)

def preprocess_topic_and_essay_dict_pretrained_wv():
    from data import ZHIHU_dataset
    from config import config_zhihu_dataset as c
    train_all_dataset = ZHIHU_dataset(c.train_data_path, c.topic_num_limit, c.essay_vocab_size, c.topic_threshold,
                                      c.topic_padding_num, c.essay_padding_len)
    pretrained_wv = read_pretrained_word_vectors(c.pretrained_wv_path)

    wv_topic = process_word_dict_and_pretrained_wv(train_all_dataset.topic2idx, pretrained_wv, c.pretrained_wv_dim)
    save_pickle_obj(wv_topic, c.topic_preprocess_wv_path)

    wv_essay = process_word_dict_and_pretrained_wv(train_all_dataset.essay2idx, pretrained_wv, c.pretrained_wv_dim)
    save_pickle_obj(wv_essay, c.essay_preprocess_wv_path)

if __name__ == '__main__':
    preprocess_topic_and_essay_dict_pretrained_wv()
