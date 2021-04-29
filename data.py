import copy
import torch
import re
import numpy as np
from torch.utils.data import Dataset
from config import config_zhihu_dataset, config_concepnet
from tools import tools_get_logger, tools_load_pickle_obj


def preprocess(sent):
    temp = []
    p = "[\u4E00-\u9FA5]+"
    for x in sent.split(' '):
        t = re.findall(p, x)
        if len(t) == 0: continue
        x = t[0]
        if x and (not x.isspace()) and (len(x) < 6):
            temp.append(x.strip())
    return temp

def load_npy(data_config):
    ret = []
    # print(data_config)
    for item in data_config:
        print(item)
        ret.append(np.load(item, allow_pickle=True))
    return ret

def read_zhihu_dataset(path, pre=True):
    essays = []
    topics = []
    with open(path, 'r', encoding='utf-8') as file:
        if pre:
            for i, line in enumerate(file):
                line = line.strip('\n').strip('\r').split('</d>')
                # process the essay
                essays.append(preprocess(line[0]))
                # process the topics
                topics.append(preprocess(line[1]))
        else:
            for i, line in enumerate(file):
                line = line.strip('\n').strip('\r').split('</d>')
                # process the essay
                essays.append(line[0].split(' '))
                # process the topics
                topics.append(line[1].split(' '))

    return essays, topics

def read_coco_dataset(path):
    essays = []
    topics = []
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            line = line.strip('\r').strip('\n').strip(' ')
            line = line.split(' ')
            essays.append(line)
            topics.append(line)
    return essays, topics

def read_acl_origin_data():
    load_path = {
        'train': './zhihu_dataset/acl_data/train.std.txt',
        'test': './zhihu_dataset/acl_data/test.std.txt',
        'valid': './zhihu_dataset/acl_data/valid.std.txt'
    }
    config = {
        "word_dict": "./zhihu_dataset/acl_data/word_dict_zhihu.npy",
        "pretrain_wv": "./zhihu_dataset/acl_data/wv_tencent.npy",
        "topic_list": "./zhihu_dataset/acl_data/topic_list_100.pkl",
        'train_mem': "./zhihu_dataset/acl_data/train_mem_idx_120_concept.npy",
        'test_mem': "./zhihu_dataset/acl_data/tst.mem.idx.120.concept.npy",
        'topic_list': './zhihu_dataset/acl_data/topic_list_100.pkl',
    }
    word2idx = np.load(config['word_dict'], allow_pickle=True).item()
    special = config_zhihu_dataset.special_tokens
    for key in special:
        word2idx[key] = word2idx[key.upper()]
        del word2idx[key.upper()]
    idx2word = {v: k for k, v in word2idx.items()}
    topic_list = tools_load_pickle_obj(config['topic_list'])
    topic2idx = {k:word2idx[k] for k in topic_list}
    idx2topic = {v:k for k, v in topic2idx.items()}
    train_essay, train_topic = read_zhihu_dataset(load_path['train'], pre=False)
    test_essay, test_topic = read_zhihu_dataset(load_path['test'], pre=False)
    train_mem, test_mem = load_npy([config['train_mem'], config['test_mem']])

    return word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem), (test_essay, test_topic, test_mem)



class ZHIHU_dataset(Dataset):
    def __init__(self, path, topic_num_limit, topic_padding_num, vocab_size, essay_padding_len, prior,
                 special_tokens=config_zhihu_dataset.special_tokens,
                 mem_corpus_path=None, encode_to_tensor=True,
                 topic_mem_per_num=config_zhihu_dataset.topic_mem_per_num,
                 topic_mem_num_all=config_zhihu_dataset.topic_mem_max_num,
                 acl_datas=None):

        self.path = path
        self.topic_num_limit = topic_num_limit
        self.vocab_size_limit = vocab_size
        self.essay_padding_len = essay_padding_len
        self.topic_padding_num = topic_padding_num
        self.data_topics = []
        self.len_topics = None
        self.len_essays = None
        self.data_essays = []
        self.data_mems = []
        self.memory_corpus = []
        self.mem_corpus_path = mem_corpus_path
        self.topic_mem_per_num = topic_mem_per_num
        self.topic_mem_num_all = topic_mem_num_all
        self.weight_for_mem_choice = np.exp([i for i in range(self.topic_mem_num_all, 0, -1)])
        self.weight_for_mem_choice /= sum(self.weight_for_mem_choice)

        if acl_datas:
            self.word2idx, self.idx2word, self.topic2idx, self.idx2topic, \
            (self.data_essays, self.data_topics, self.data_mems) = acl_datas
            self.data_mems = torch.from_numpy(self.data_mems)
        else:
            if prior:
                self.word2idx, self.idx2word, self.topic2idx, self.idx2topic, self.memory_corpus = \
                    prior['word2idx'], prior['idx2word'], prior['topic2idx'], prior['idx2topic'], prior['memory_corpus']
                self.data_essays, self.data_topics, _, _, _, _ = self.__read_datas(special_tokens, has_prior=True)
            else:
                self.data_essays, self.data_topics, self.word2idx, self.idx2word, self.topic2idx, self.idx2topic = \
                    self.__read_datas(special_tokens)

        if encode_to_tensor:
            self.__encode_datas()



    def __read_datas(self, speicial_tokens:set, has_prior=False):
        # return essays, topics, word2idx, idx2word
        tools_get_logger('data').info(f'reading datas from {self.path}')
        if 'zhihu' in self.path.lower():
            essays, topics = read_zhihu_dataset(self.path)
        elif 'coco' in self.path.lower():
            essays, topics = read_coco_dataset(self.path)
        else:
            raise NotImplementedError(f"{self.path} not supported")


        if has_prior:
            return essays, topics, None, None, None, None


        inf = int(1e9)
        word_cnts = {k: inf + 10 for k in speicial_tokens}  # force reserve
        topic_cnts = {}

        for ee in essays:
            for w in ee:
                if w not in word_cnts:
                    word_cnts[w] = 0
                word_cnts[w] += 1

        for tt in topics:
            for t in tt:
                if t not in topic_cnts:
                    topic_cnts[t] = 0
                topic_cnts[t] += 1


        topics_dict, topics_cnts = self.__limit_size_by_frequency(topic_cnts, reserved_num=self.topic_num_limit)
        for t in topics_dict.keys(): word_cnts[t] = inf

        if self.mem_corpus_path:
            self.memory_corpus = tools_load_pickle_obj(self.mem_corpus_path)
            # {topic: [list of commonsense]}
            for t in topics_dict.keys():
                for mem in self.memory_corpus[t]:
                    if mem not in word_cnts:
                        word_cnts[mem] = 1
                    word_cnts[mem] *= (1.5 * topics_cnts[t]) # not inf but high weighted

        word2idx = {}
        if len(word_cnts) > self.vocab_size_limit:
            word2idx, word_cnts = self.__limit_size_by_frequency(word_cnts, reserved_num=self.vocab_size_limit)
        else:
            for k in word_cnts.keys():
                word2idx[k] = len(word2idx)

        idx2word = {v:k for k, v in word2idx.items()}
        topic2idx = {t:word2idx[t] for t in topics_dict.keys()}
        idx2topic = {v:k for k, v in topic2idx.items()}



        return essays, topics, word2idx, idx2word, topic2idx, idx2topic

    def __limit_size_by_frequency(self, cnts:dict, low=0.0, high_top=30, reserved_num=None):
        cnts = sorted(cnts.items(), key=lambda item: item[1], reverse=True)
        words = {}
        if reserved_num and isinstance(reserved_num, int):
            for k, v in cnts[:reserved_num]:
                words[k] = len(words)
        else:
            low = int(len(cnts) * low)
            assert low + high_top < len(cnts)
            if low > 0: cnts = cnts[:-low]
            for k, v in cnts[high_top:]:
                words[k] = len(words)

        cnts = dict(cnts)
        cnts = {k:cnts[k] for k in words.keys()}

        return words, cnts

    def limit_datas(self, topic_threshold, essay_min_len):
        # delete the data whose the number of in dict topics below the threshold
        # using before encode_datas
        assert self.len_essays == None
        delete_indexs = []
        for i, (dt, de) in enumerate(zip(self.data_topics, self.data_essays)):
            cnt = 0
            for t in dt:
                cnt += (t in self.topic2idx)
            if cnt < topic_threshold or len(de) < essay_min_len:
                delete_indexs.append(i)

        for d in sorted(delete_indexs, reverse=True):
            del self.data_essays[d]
            del self.data_topics[d]

        tools_get_logger('data').info(f'delete {len(delete_indexs)}')

        return delete_indexs

    def __encode_datas(self):
        assert len(self.data_topics) == len(self.data_essays)
        self.len_topics = [0 for _ in self.data_topics]
        self.len_essays = [0 for _ in self.data_essays]
        essays = {'input':[], 'target': []}
        for i, (t, e) in enumerate(zip(self.data_topics, self.data_essays)):
            self.data_topics[i], self.len_topics[i] = self.convert_topic2idx(t, ret_tensor=True)
            ei, et, self.len_essays[i] = self.convert_word2idx(e, ret_tensor=True)
            essays['input'].append(ei)
            essays['target'].append(et)
        self.data_essays = essays
        if self.memory_corpus and len(self.memory_corpus) > 0:
            self.data_mems = self.shuffle_memory()
        # limit = 1000
        # self.data_topics = self.data_topics[:limit]
        # self.len_topics = self.len_topics[:limit]
        # self.data_mems = self.data_mems[:limit]
        # self.data_essays['input'] = self.data_essays['input'][:limit]
        # self.data_essays['target'] = self.data_essays['target'][:limit]
        # self.len_essays = self.len_essays[:limit]


    def convert_idx2word(self, idxs, sep=False, end_token=None):
        temp = []
        if isinstance(end_token, str):
            end_token = self.word2idx[end_token]
        for i in idxs:
            if i == end_token: break
            temp.append(self.idx2word[i])
        if isinstance(sep, str):
            return sep.join(temp)
        return temp

    def unpadded_idxs(self, idxs, end_token='<eos>'):
        end_token = self.word2idx[end_token]
        temp = []
        for i in idxs:
            if i == end_token: break
            temp.append(i)
        return temp


    def convert_idx2topic(self, idxs):
        return [self.idx2word[i] for i in idxs]

    def convert_topic2idx(self, topics, ret_tensor):
        # will remove all the unk topic and shift them to the left then padding to limit
        padding_num = self.topic_padding_num
        temp = [self.word2idx['<pad>'] for _ in range(padding_num)]
        real_num = 0
        for i, one in enumerate(topics[:padding_num]):
            if one in self.topic2idx:
                temp[real_num] = self.topic2idx[one]
                real_num += 1
        if ret_tensor:
            return torch.tensor(temp, dtype=torch.int64), torch.tensor(real_num, dtype=torch.int64)
        return temp, real_num

    def convert_word2idx(self, words, ret_tensor=False):
        # return (essay_input, essay_target, essay_real_len)
        # essay_input <go> words <pad>
        # essay_target words <eos> <pad>
        padding_len = self.essay_padding_len
        temp = [self.word2idx['<pad>'] for _ in range(padding_len+1)]
        temp[0] = self.word2idx['<go>']
        real_len = min(len(words), padding_len - 1)
        for i, one in enumerate(words[:padding_len]):
            if one not in self.word2idx:
                temp[i+1] = self.word2idx['<unk>']
            else:
                temp[i+1] = self.word2idx[one]
        essay_input = temp[:-1]
        essay_target = temp[1:]
        essay_target[real_len] = self.word2idx['<eos>']
        if ret_tensor:
            return torch.tensor(essay_input, dtype=torch.int64), torch.tensor(essay_target, dtype=torch.int64), \
            torch.tensor(real_len, dtype=torch.int64)
        return essay_input, essay_target, real_len

    def convert_mem2idx(self, mems, ret_tensor=False):
        temp = [self.word2idx['<unk>'] for _ in mems]
        for i, m in enumerate(mems):
            if m in self.word2idx: temp[i] = self.word2idx[m]

        if ret_tensor:
            return torch.tensor(temp, dtype=torch.int64)
        return temp

    def shuffle_memory(self):
        data_mems = [None for _ in self.data_topics]
        for i in range(len(self)):
            # because only convert it to idxs and then reverse it to words
            # which will let unk_topic_word be encoded to <unk> that memory_corpus knows
            temp = self.convert_idx2topic(self.data_topics[i].tolist())
            data_mems[i] = self.get_mems_by_topics(temp, ret_tensor=True)
        return data_mems

    def get_mems_by_topics(self, topics:list, ret_tensor=False):
        # topics contains series of topic_words
        mems = ['<unk>' for _ in range(self.topic_mem_num_all)]
        has_set = set()
        for t in topics:
            if t in self.memory_corpus and self.memory_corpus[t][0] != '<unk>':
                has_set.add(t)
        if len(has_set) != 0:
            mems = []
            nums = [self.topic_mem_num_all // len(has_set) for _ in has_set]
            if self.topic_mem_num_all % len(has_set) != 0:
                nums[0] = self.topic_mem_num_all - sum(nums[1:])
            for t, n in zip(has_set, nums):
                temp = np.random.choice(self.memory_corpus[t], n, replace=False, p=self.weight_for_mem_choice)
                mems.extend(temp)

        return self.convert_mem2idx(mems, ret_tensor=ret_tensor)

    def print_info(self):
        tools_get_logger('data').info(
            f"data_num {len(self)} topic_num/topic_num_limit {len(self.topic2idx)}/{self.topic_num_limit} \n"
            f"vocab_size/vocab_size_limit {len(self.word2idx)}/{self.vocab_size_limit} \n"
            f"topic_pad_num {self.topic_padding_num} essay_padding_len {self.essay_padding_len} \n"
            f"load_memory_path {self.mem_corpus_path} mem_corpus_num {len(self.memory_corpus)} \n"
        )

    def get_prior(self):
        return {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'topic2idx': self.topic2idx,
            'idx2topic': self.idx2topic,
            'memory_corpus': self.memory_corpus
        }

    def __len__(self):
        return len(self.data_topics)

    def __getitem__(self, item):
        if len(self.data_mems) == 0:
            return self.data_essays['input'][item], self.data_essays['target'][item], self.data_topics[item]
        return (self.data_topics[item], self.len_topics[item], self.data_mems[item],
                self.data_essays['input'][item], self.data_essays['target'][item], self.len_essays[item])

    def __setitem__(self, key, value):
        if len(self.data_mems) == 0:
            self.data_essays['input'][key], self.data_essays['target'][key], self.data_topics[key] = value
        self.data_topics[key], self.len_topics[key], self.data_mems[key], self.data_essays['input'][key], \
        self.data_essays['target'][key], self.len_essays[key] = value

class InputLabel_dataset(Dataset):
    def __init__(self, inputs:list, labels:list, dtypes={'input': torch.int64, 'label': torch.float}):
        assert len(inputs) == len(labels)
        if not isinstance(inputs[0], torch.Tensor):
            inputs = [torch.tensor(x, dtype=dtypes['input']) for x in inputs]
        if not isinstance(labels[0], torch.Tensor):
            labels = [torch.tensor(l, dtype=dtypes['label']) for l in labels]

        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.labels[item]

    def __setitem__(self, key, value):
        self.inputs[key], self.labels[key] = value




if __name__ == '__main__':
    pass
    word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem), (test_essay, test_topic, test_mem) \
    = read_acl_origin_data()
    train_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
                                      topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                      topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                      vocab_size=config_zhihu_dataset.vocab_size,
                                      essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                      prior=None, encode_to_tensor=True,
                                      mem_corpus_path=config_concepnet.memory_corpus_path,
                                      acl_datas=(word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem)))
    train_all_dataset.print_info()


    pass





