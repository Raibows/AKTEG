

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


def k_fold_split(all_dataset, batch_size, k=5):
    """
    if k==1, then all_dataset will set to train_dataset, and corresponding validation_dataset is set to None
    else
    every fold contains a different
    validation_dataset with size (num_all // k) and a train_dataset with size (num_all - num_all // k)
    """
    import random
    import torch
    from torch.utils.data import DataLoader
    from config import config_train
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
        if k == 1:
            kfolds.append((DataLoader(all_dataset, config_train.batch_size, shuffle=True,
                                      num_workers=config_train.dataloader_num_workers, pin_memory=True), None))
            return kfolds
        test = torch.utils.data.dataset.Subset(all_dataset, list(fs))
        train = torch.utils.data.dataset.Subset(all_dataset, list(all_set-fs))
        kfolds.append((
            DataLoader(train, batch_size=batch_size, num_workers=config_train.dataloader_num_workers,
                       shuffle=True, pin_memory=True),
            DataLoader(test, batch_size=batch_size, num_workers=config_train.dataloader_num_workers, pin_memory=True)
        ))
    return kfolds

def read_pretrained_word_vectors(path, wv_dim):
    from tools import tools_get_logger
    from tqdm import tqdm
    tools_get_logger('preprocess').info(f"loading pretrained word vectors from {path}")
    pretrained_wv = {}
    if 'tencent' in path.lower():
        total = 8824331
        desc = 'tencent_wv'
        skip = 0
    else:
        total = 259922
        desc = 'zhihu_wv'
        skip = -10
    with open(path, 'r', encoding='utf-8') as file:
        with tqdm(total=total, desc=desc) as pbar:
            for i, line in enumerate(file):
                pbar.update(1)
                if i == skip: continue
                line = line.strip('\n\r').split(' ')
                wv = list(map(lambda x: float(x), line[1:]))
                assert len(wv) == wv_dim
                pretrained_wv[line[0]] = wv
    return pretrained_wv

def process_word_dict_and_pretrained_wv(word_dict:dict, pretrained_wv: dict, wv_dim: int, unk_val='gaussian'):
    from tools import tools_setup_seed
    import random
    tools_setup_seed(667)
    if unk_val == 'gaussian':
        wv = [[random.normalvariate(0, 0.3) for _ in range(wv_dim)] for i in word_dict]
    else:
        wv = [[unk_val for _ in range(wv_dim)] for i in word_dict]
    unk_num = 0
    for k, v in word_dict.items():
        if k in pretrained_wv: wv[v] = pretrained_wv[k]
        else: unk_num += 1
    print(f'unk_num {unk_num} finding {len(word_dict)-unk_num} all {len(word_dict)}')
    return wv

def generate_pretrained_wv(pretrained_wv=None):
    # make sure you have build latest memory before
    from data import ZHIHU_dataset
    from config import config_zhihu_dataset as c
    from config import config_seq2seq as s
    from config import config_concepnet
    from tools import tools_save_pickle_obj

    name = 'tencent'
    assert config_concepnet.memory_corpus_path != None
    train_all_dataset = ZHIHU_dataset(c.train_data_path, topic_num_limit=c.topic_num_limit,
                                      topic_padding_num=c.topic_padding_num, vocab_size=c.vocab_size,
                                      essay_padding_len=c.essay_padding_len, prior=None, encode_to_tensor=False,
                                      mem_corpus_path=config_concepnet.memory_corpus_path)
    if not pretrained_wv:
        pretrained_wv = read_pretrained_word_vectors(c.pretrained_wv_path[name], c.pretrained_wv_dim[name])

    wv = process_word_dict_and_pretrained_wv(train_all_dataset.word2idx, pretrained_wv, c.pretrained_wv_dim[name])

    tools_save_pickle_obj(wv, s.pretrained_wv_path[name])

def build_commonsense_memory():
    from data import ZHIHU_dataset
    from config import config_zhihu_dataset as cz
    from config import config_concepnet as cc
    import random
    import synonyms
    from tools import tools_save_pickle_obj, tools_load_pickle_obj, tools_setup_seed
    import time

    tools_setup_seed(667)
    train_all_dataset = ZHIHU_dataset(path=cz.train_data_path, topic_num_limit=cz.topic_num_limit,
                                      topic_padding_num=cz.topic_padding_num, vocab_size=cz.vocab_size,
                                      essay_padding_len=cz.essay_padding_len, prior=None, mem_corpus_path=None)
    wv_name = 'tencent'
    pretrained_wv = read_pretrained_word_vectors(cz.pretrained_wv_path[wv_name], cz.pretrained_wv_dim[wv_name])
    concepnet_dict = tools_load_pickle_obj(cc.reserved_data_path)
    # topic_memory_corpus = [['<oov>' for i in range(cz.topic_mem_max_num)] for j in range(train_all_dataset.topic_num_limit)]
    topic_memory_corpus = {} # {topic_word : [list of synonyms...]}

    for k, v in train_all_dataset.topic2idx.items():
        synonym_add_num = cz.topic_mem_max_num
        topic_memory_corpus[k] = ['<unk>' for _ in range(cz.topic_mem_max_num)]
        if k in concepnet_dict:
            candidates = concepnet_dict[k]
            if len(candidates) > cz.topic_mem_max_num:
                candidates = random.sample(candidates, k=cz.topic_mem_max_num)
            for ii, one in enumerate(candidates): topic_memory_corpus[k][ii] = one
            synonym_add_num -= len(candidates)

        if synonym_add_num <= 0: continue
        # let synonyms to replenish to max_num
        # multiply by 2 is to make sure having enough synonyms in pretrained_wv
        temp = synonyms.nearby(k, synonym_add_num * 2)[0]
        i = cz.topic_mem_max_num - synonym_add_num
        flag = 0
        for j, one in enumerate(temp):
            if one in pretrained_wv:
                topic_memory_corpus[k][i] = one
                i += 1
                flag = j
            if i >= cz.topic_mem_max_num: break
        flag += 1
        while i < cz.topic_mem_max_num and flag < len(temp):
            topic_memory_corpus[k][i] = temp[flag]
            i += 1
            flag += 1

    tools_save_pickle_obj(topic_memory_corpus, cc.memory_corpus_path)


def split_train_test_set():
    from data import ZHIHU_dataset
    from config import config_zhihu_dataset
    from tools import tools_get_logger
    raw_dataset = ZHIHU_dataset(path=config_zhihu_dataset.raw_data_path,
                                topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                vocab_size=config_zhihu_dataset.vocab_size,
                                essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                prior=None, encode_to_tensor=False)

    delete_indexs = raw_dataset.limit_datas(topic_threshold=config_zhihu_dataset.preprocess_topic_threshold,
                                            essay_min_len=config_zhihu_dataset.preprocess_essay_min_len)

    datas = read_line_like_file(config_zhihu_dataset.raw_data_path)
    all = set([i for i in range(len(datas))]) - set(delete_indexs)
    datas = [datas[i] for i in all]
    train_datas, test_datas = split_line_like_datas(datas, config_zhihu_dataset.test_data_split_ratio)
    write_line_like_file(config_zhihu_dataset.train_data_path, train_datas)
    write_line_like_file(config_zhihu_dataset.test_data_path, test_datas)
    tools_get_logger('preprocess').info(f'train dataset num {len(train_datas)} test dataset num {len(test_datas)} '
                                   f'test / train is {len(test_datas)/len(train_datas):.4f} '
                                   f'test / (train+test) is {len(test_datas)/len(all):.4f}')

def preprocess_concepnet():
    from config import config_concepnet as c
    from tools import tools_get_logger, tools_save_pickle_obj
    from zhconv import convert
    import re

    with open(c.raw_path, 'r', encoding='utf-8') as file:
        total = 0
        pattern = "[^\u4e00-\u9fa5]"
        reserved = {}
        for i, line in enumerate(file):
            total += 1
            line = line.split('\t')
            src = line[2]
            tar = line[3]
            if (not src.startswith('/c/zh/')) or (not tar.startswith('/c/zh/')): continue
            src = convert(src[6:], locale='zh-cn')
            tar = convert(tar[6:], locale='zh-cn')
            src = re.sub(pattern, "", src)
            tar = re.sub(pattern, "", tar)
            if src not in reserved:
                reserved[src] = set()
            reserved[src].add(tar)
    reserved = {k: list(v) for k, v in reserved.items()}
    num = sum([len(t) for t in reserved.values()])
    tools_get_logger('preprocess').info(f"read concepnet from {c.raw_path} reserved/total {num}/{total} \n"
                                        f"reserved is writing to {c.reserved_data_path}")

    # {word: [syn1, syn2, syn3...], word2: [...]}
    tools_save_pickle_obj(reserved, c.reserved_data_path)





if __name__ == '__main__':
    # split_train_test_set()
    # build_commonsense_memory()
    generate_pretrained_wv()
    # preprocess_concepnet()
    pass