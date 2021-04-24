import copy
import torch
import re
import numpy as np
from torch.utils.data import Dataset
from config import config_zhihu_dataset, config_concepnet
from tools import tools_get_logger, tools_load_pickle_obj


import random


class ZHIHU_dataset(Dataset):
    def __init__(self, path, topic_num_limit, essay_vocab_size, topic_threshold, topic_padding_num, essay_padding_len,
                 topic_special_tokens=config_zhihu_dataset.topic_special_tokens,
                 essay_special_tokens=config_zhihu_dataset.essay_special_tokens,
                 prior=None, load_mems=True, encode_to_tensor=True):

        self.path = path
        self.topic_num_limit = topic_num_limit
        self.essay_vocab_size = essay_vocab_size
        self.topic_threshold = topic_threshold
        self.essay_padding_len = essay_padding_len
        self.topic_padding_num = topic_padding_num
        self.data_topics = None
        self.len_topics = None
        self.len_essays = None
        self.data_essays = None
        self.data_mems = None
        self.delete_indexs = []
        self.mem2idx = None
        self.idx2mem = None
        self.memory_corpus = None
        self.load_mems = load_mems
        self.topic_mem_normal_num = config_zhihu_dataset.topic_mem_normal_num
        self.topic_mem_max_num = config_zhihu_dataset.topic_mem_max_num
        self.weight_for_mem_choice = np.exp([i for i in range(self.topic_mem_max_num, 0, -1)])
        self.weight_for_mem_choice /= sum(self.weight_for_mem_choice)

        temp_topic2idx, temp_essay2idx, self.data_topics, self.data_essays = \
            self.__read_datas(essay_special_tokens, topic_special_tokens)



        if not prior:
            self.topic2idx, self.idx2topic = \
                self.__limit_dict_by_frequency(topic_special_tokens, temp_topic2idx,
                                               self.topic_num_limit, self.data_topics,
                                               remove_high_top=0)
            self.essay2idx, self.idx2essay = \
                self.__limit_dict_by_frequency(essay_special_tokens, temp_essay2idx,
                                               self.essay_vocab_size, self.data_essays,
                                               remove_high_top=config_zhihu_dataset.remove_high_freq_top)
            if load_mems:
                self.mem2idx, self.idx2mem = tools_load_pickle_obj(config_concepnet.mem2idx_and_idx2mem_path)
                self.memory_corpus = tools_load_pickle_obj(config_concepnet.topic_2_mems_corpus_path)
            self.word2idx, self.idx2word = self.merge_temp2idx_to_reserved({})

        else:
            self.topic2idx, self.idx2topic = prior['topic2idx'], prior['idx2topic']
            self.essay2idx, self.idx2essay = prior['essay2idx'], prior['idx2essay']
            if load_mems:
                self.mem2idx, self.idx2mem = prior['mem2idx'], prior['idx2mem']
                self.memory_corpus = prior['memory_corpus']
            self.word2idx, self.idx2word = prior['word2idx'], prior['idx2word']

        self.essay_vocab_size = min(len(self.essay2idx), self.essay_vocab_size)
        self.topic_num_limit = min(len(self.topic2idx), self.topic_num_limit)
        self.mem_vocab_size = len(self.mem2idx) if self.load_mems else 0
        self.word_vocab_size = len(self.word2idx)


        if encode_to_tensor:
            self.__encode_datas()
        self.print_info()


    def __encode_datas(self):
        assert len(self.data_topics) == len(self.data_essays)
        self.len_topics = [0 for _ in self.data_topics]
        self.len_essays = [0 for _ in self.data_essays]
        essays = {'input':[], 'target': []}
        self.data_mems = [0 for _ in self.data_topics]
        for i, (t, e) in enumerate(zip(self.data_topics, self.data_essays)):
            self.data_topics[i], self.len_topics[i] = self.convert_word2idx(t, special='topic', ret_tensor=True)
            if self.load_mems:
                # because only convert it to idxs and then reverse it to words
                # which will let unk_topic_word be encoded to <unk_topic> that memory_corpus knows
                temp = self.convert_idx2words(self.data_topics[i].tolist(), special='topic')
                self.data_mems[i] = self.get_mems_by_topics(temp, ret_tensor=True)
            ei, et, self.len_essays[i] = self.convert_word2idx(e, special='essay', ret_tensor=True)
            essays['input'].append(ei)
            essays['target'].append(et)
        self.data_essays = essays

    def __limit_dict_by_frequency(self, reserved, temp2idx, size, datas, remove_high_top=100):
        assert datas and size >= len(reserved)
        # update reserved
        for k in reserved.keys():
            if k not in temp2idx:
                temp2idx[k] = len(temp2idx)
        idx2temp = [None for _ in temp2idx]
        for k, v in temp2idx.items(): idx2temp[v] = k
        cnts = [[i, 0] for i in idx2temp]
        for v in reserved.values(): cnts[v][1] = int(1e9)
        for one in datas:
            for word in one: cnts[temp2idx[word]][1] += 1
        cnts = sorted(cnts, reverse=True, key=lambda t: t[1])
        reserved.clear()
        for i in range(len(cnts)):
            if cnts[i][1] == int(1e9) or i >= remove_high_top:
                reserved[cnts[i][0]] = len(reserved)
            if len(reserved) == size: break
        idx2temp = [None for _ in reserved]
        for k, v in reserved.items(): idx2temp[v] = k

        return reserved, idx2temp

    def limit_datas(self):
        # delete the data whose the number of in dict topics below the threshold
        # using before encode_datas
        assert self.len_essays == None
        delete_indexs = []
        for i, (dt, de) in enumerate(zip(self.data_topics, self.data_essays)):
            cnt = 0
            for t in dt:
                cnt += (t in self.topic2idx)
            if cnt < self.topic_threshold or len(de) < self.essay_padding_len // 2:
                delete_indexs.append(i)

        for d in sorted(delete_indexs, reverse=True):
            del self.data_essays[d]
            del self.data_topics[d]
        assert len(self.data_essays) == len(self.data_topics)
        tools_get_logger('data').info(f'delete {len(delete_indexs)}')
        self.delete_indexs = delete_indexs
        return delete_indexs

    def merge_temp2idx_to_reserved(self, reserved:dict):
        # merge topic2idx and mem2idx(optional) to exists essay2idx to expand the dict
        # the expand dict will in reserved could keep reserved in follow limit_dict operation
        assert len(self.topic2idx) > 0
        assert len(self.essay2idx) > 0

        for k, v in self.topic2idx.items():
            if k not in reserved: reserved[k] = len(reserved)
            self.topic2idx[k] = reserved[k]
        self.idx2topic = {v:k for k, v in self.topic2idx.items()}

        for k, v in self.essay2idx.items():
            if k not in reserved: reserved[k] = len(reserved)
            self.essay2idx[k] = reserved[k]
        self.idx2essay = {v:k for k, v in self.essay2idx.items()}


        if self.mem2idx:
            for k, v in self.mem2idx.items():
                if k not in reserved: reserved[k] = len(reserved)
                self.mem2idx[k] = reserved[k]
            self.idx2mem = {v:k for k, v in self.mem2idx.items()}

        return reserved, {v:k for k, v in reserved.items()}

    def __preprocess(self, sent):
        temp = []
        p = "[\u4E00-\u9FA5]+"
        for x in sent.split(' '):
            t = re.findall(p, x)
            if len(t) == 0: continue
            x = t[0]
            if x and (not x.isspace()) and (len(x) < 5):
                temp.append(x.strip())
        return temp

    def __read_datas(self, essay2idx, topic2idx, limit_num=-1):
        topics = []
        essays = []
        essay2idx = copy.deepcopy(essay2idx)
        topic2idx = copy.deepcopy(topic2idx)
        with open(self.path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i == limit_num: break
                line = line.strip('\n').strip('\r').split('</d>')
                # process the essay
                temp = self.__preprocess(line[0])
                for l in temp:
                    if l not in essay2idx: essay2idx[l] = len(essay2idx)
                essays.append(temp)
                # process the topics
                temp = self.__preprocess(line[1])
                for t in temp:
                    if t not in topic2idx: topic2idx[t] = len(topic2idx)
                topics.append(temp)



        tools_get_logger('data').info(f'read origin data {len(topics)} from {self.path}')

        return topic2idx, essay2idx, topics, essays

    def convert_idx2words(self, idxs, special=None):
        if special == 'topic':
            return [self.idx2topic[i] for i in idxs]
        if special == 'essay':
            return [self.idx2essay[i] for i in idxs]
        if special == 'mem':
            return [self.idx2mem[i] for i in idxs]
        return [self.idx2word[i] for i in idxs]

    def convert_word2idx(self, words, special, ret_tensor=False):
        if special == 'essay':
            return self.__convert_essay2idx(words, ret_tensor=ret_tensor)
        if special == 'topic':
            return self.__convert_topic2idx(words, ret_tensor=ret_tensor)
        if special == 'mem':
            return self.__convert_mem2idx(words, ret_tensor=ret_tensor)
        raise KeyError(f'speicl {special} must in essay, topic or mem')

    def get_mems_by_topics(self, topics:list, ret_tensor=False):
        # topics contains series of topic_words
        mems = ['<oov>' for _ in range(self.topic_mem_max_num)]
        has_set = set()
        for t in topics:
            if self.memory_corpus[t][0] != '<oov>':
                has_set.add(t)
        if len(has_set) != 0:
            mems = []
            nums = [self.topic_mem_max_num // len(has_set) for _ in has_set]
            if self.topic_mem_max_num % len(has_set) != 0:
                nums[0] = self.topic_mem_max_num - sum(nums[1:])
            for t, n in zip(has_set, nums):
                temp = np.random.choice(self.memory_corpus[t], n, replace=False, p=self.weight_for_mem_choice)
                mems.extend(temp)

        return self.convert_word2idx(mems, special='mem', ret_tensor=ret_tensor)

    def shuffle_memory(self):
        for i in range(len(self)):
            temp = self.convert_idx2words(self.data_topics[i].tolist(), special='topic')
            self.data_mems[i] = self.get_mems_by_topics(temp, ret_tensor=True)

    def __convert_mem2idx(self, mems, ret_tensor=False):
        temp = [self.word2idx[i] for i in mems]
        if ret_tensor: return torch.tensor(temp, dtype=torch.int64)
        return temp

    def __convert_essay2idx(self, essay, padding_len=None, ret_tensor=False):
        # return (essay_input, essay_target, essay_real_len)
        # essay_input <sos> words <pad>
        # essay_target words <eos> <pad>
        if padding_len == None:
            padding_len = self.essay_padding_len
        temp = [self.word2idx['<pad>'] for _ in range(padding_len+1)]
        temp[0] = self.word2idx['<sos>']
        real_len = min(len(essay), padding_len - 1)
        for i, one in enumerate(essay[:padding_len]):
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

    def __convert_topic2idx(self, topic, padding_num=None, ret_tensor=False):
        # returned real_num not containing the <eos_topic>
        # so you may need to +1 manually
        if padding_num == None:
            padding_num = self.topic_padding_num
        temp = [self.topic2idx['<pad_topic>'] for _ in range(padding_num)]
        real_num = min(len(topic), padding_num - 1)
        for i, one in enumerate(topic[:padding_num]):
            if one not in self.topic2idx:
                temp[i] = self.topic2idx['<unk_topic>']
            else:
                temp[i] = self.topic2idx[one]
        temp[real_num] = self.topic2idx['<eos_topic>']
        if ret_tensor:
            return torch.tensor(temp, dtype=torch.int64), torch.tensor(real_num, dtype=torch.int64)
        return temp, real_num

    def print_info(self):
        assert len(self.essay2idx) == self.essay_vocab_size
        assert len(self.topic2idx) == self.topic_num_limit
        tools_get_logger('data').info(
            f"data_num {len(self)} topic_num/topic_num_limit {len(self.topic2idx)}/{self.topic_num_limit} \n"
            f"essay_vocab/essay_vocab_limit {len(self.essay2idx)}/{self.essay_vocab_size} \n"
            f"topic_pad_num {self.topic_padding_num} essay_padding_len {self.essay_padding_len} \n"
            f"is_load_memory {self.load_mems} mem_num {self.mem_vocab_size} \n"
            f"embedding_size {len(self.word2idx)} data_delete_num {len(self.delete_indexs)}"
        )

    def get_prior(self):
        return {'topic2idx': self.topic2idx,
                'idx2topic': self.idx2topic,
                'essay2idx': self.essay2idx,
                'idx2essay': self.idx2essay,
                'mem2idx': self.mem2idx,
                'idx2mem': self.idx2mem,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'memory_corpus': self.memory_corpus}

    def __len__(self):
        return len(self.data_topics)

    def __getitem__(self, item):
        return (self.data_topics[item], self.len_topics[item], self.data_mems[item],
                self.data_essays['input'][item], self.data_essays['target'][item], self.len_essays[item])

    def __setitem__(self, key, value):
        self.data_topics[key], self.len_topics[key], self.data_mems[key], self.data_essays['input'][key], \
        self.data_essays['target'][key], self.len_essays[key] = value



if __name__ == '__main__':
    # essay_special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3, }
    pass









