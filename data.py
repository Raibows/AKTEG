import copy
import torch
from torch.utils.data import Dataset
from config import config_zhihu_dataset, config_concepnet
from tools import tools_get_logger
from tools import tools_load_pickle_obj


class ZHIHU_dataset(Dataset):
    def __init__(self, path, topic_num_limit, essay_vocab_size, topic_threshold, topic_padding_num, essay_padding_len,
                 topic_special_tokens=config_zhihu_dataset.topic_special_tokens,
                 essay_special_tokens=config_zhihu_dataset.essay_special_tokens,
                 prior=None):

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
        self.topic_synonym_normal_num = config_zhihu_dataset.topic_mem_normal_num
        self.topic_synonym_max_num = config_zhihu_dataset.topic_mem_max_num

        temp_topic2idx, temp_essay2idx, self.data_topics, self.data_essays = \
            self.__read_datas(essay_special_tokens, topic_special_tokens)

        self.topic_num_limit = min(self.topic_num_limit, len(temp_topic2idx))
        self.essay_vocab_size = min(self.essay_vocab_size, len(temp_essay2idx))

        if not prior:
            self.topic2idx, self.idx2topic = self.__limit_dict_by_frequency(topic_special_tokens, temp_topic2idx,
                                                                            self.topic_num_limit, self.data_topics)
            self.essay2idx, self.idx2essay = self.__limit_dict_by_frequency(essay_special_tokens, temp_essay2idx,
                                                                            self.essay_vocab_size, self.data_essays)
            self.mem2idx, self.idx2mem = tools_load_pickle_obj(config_concepnet.mem2idx_and_idx2mem_path)
            self.memory_corpus = tools_load_pickle_obj(config_concepnet.topic_2_mems_corpus_path)
        else:
            self.topic2idx, self.idx2topic = prior['topic2idx'], prior['idx2topic']
            self.essay2idx, self.idx2essay = prior['essay2idx'], prior['idx2essay']
            self.mem2idx, self.idx2mem = prior['mem2idx'], prior['idx2mem']
            self.memory_corpus = prior['memory_corpus']

        self.mem_vocab_size = len(self.mem2idx)

        self.__encode_datas()
        self.print_info()


    def __encode_datas(self):
        assert len(self.data_topics) == len(self.data_essays)
        self.len_topics = [0 for _ in self.data_topics]
        self.len_essays = [0 for _ in self.data_essays]
        essays = {'input':[], 'target': []}
        self.data_mems = [0 for _ in self.data_topics]
        for i, (t, e) in enumerate(zip(self.data_topics, self.data_essays)):
            self.data_topics[i], self.len_topics[i] = self.convert_topic2idx(t, ret_tensor=True)
            self.data_mems[i] = self.get_mems_by_topics(self.data_topics[i].tolist(), ret_tensor=True)
            ei, et, self.len_essays[i] = self.convert_essay2idx(e, ret_tensor=True)
            essays['input'].append(ei)
            essays['target'].append(et)
        self.data_essays = essays

    def __limit_dict_by_frequency(self, reserved, temp2idx, size, datas):
        assert datas and size >= len(reserved)
        idx2temp = [None for _ in temp2idx]
        for k, v in temp2idx.items(): idx2temp[v] = k
        cnts = [[i, 0] for i in idx2temp]
        for v in reserved.values(): cnts[v][1] = int(1e9)
        for one in datas:
            for word in one: cnts[temp2idx[word]][1] += 1
        cnts = sorted(cnts, reverse=True, key=lambda t: t[1])
        reserved.clear()
        for i in range(size): reserved[cnts[i][0]] = len(reserved)
        idx2temp = [None for _ in reserved]
        for k, v in reserved.items(): idx2temp[v] = k

        return reserved, idx2temp

    def limit_datas(self):
        # delete the data whose the number of in dict topics below the threshold
        delete_indexs = []
        for i, one in enumerate(self.data_topics):
            cnt = 0
            for t in one:
                cnt += (t in self.essay2idx)
            if cnt < self.topic_threshold:
                delete_indexs.append(i)

        for d in sorted(delete_indexs, reverse=True):
            del self.data_essays[d]
            del self.data_topics[d]
        assert len(self.data_essays) == len(self.data_topics)
        tools_get_logger('data').info(f'delete {len(delete_indexs)}')
        self.delete_indexs = delete_indexs
        return delete_indexs

    def __preprocess(self, sent):
        temp = []
        for x in sent.split(' '):
            if x and (not x.isspace()):
                temp.append(x.strip())
        return temp

    def __read_datas(self, essay2idx, topic2idx):
        topics = []
        essays = []
        essay2idx = copy.deepcopy(essay2idx)
        topic2idx = copy.deepcopy(topic2idx)
        with open(self.path, 'r', encoding='utf-8') as file:
            for line in file:
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

    def convert_idx2essay(self, idxs):
        return [self.idx2essay[i] for i in idxs]

    def convert_idx2topic(self, idxs):
        return [self.idx2topic[i] for i in idxs]

    def convert_idx2mem(self, idxs):
        return [self.idx2mem[i] for i in idxs]

    def convert_mem2idx(self, mems):
        temp = [self.mem2idx[i] for i in mems]
        return temp

    def get_mems_by_topics(self, topics:list, ret_tensor=False):
        # topics contains series of topic_idxs
        mems = ['<oov>' for _ in range(self.topic_synonym_max_num)]
        has_set = set()
        for t in topics:
            if self.memory_corpus[t][0] != '<oov>':
                has_set.add(t)
        if len(has_set) != 0:
            mems = []
            nums = [self.topic_synonym_max_num // len(has_set) for _ in has_set]
            if self.topic_synonym_max_num % len(has_set) != 0:
                nums[0] = self.topic_synonym_max_num - sum(nums[1:])
            for t, n in zip(has_set, nums):
                mems.extend(self.memory_corpus[t][:n])

        mems = self.convert_mem2idx(mems)

        if ret_tensor:
            return torch.tensor(mems, dtype=torch.int64)
        return mems

    def convert_essay2idx(self, essay, padding_len=None, ret_tensor=False):
        # return (essay_input, essay_target, essay_real_len)
        # essay_input <sos> words <pad>
        # essay_target words <eos> <pad>
        if padding_len == None:
            padding_len = self.essay_padding_len
        temp = [self.essay2idx['<pad>'] for _ in range(padding_len+1)]
        temp[0] = self.essay2idx['<sos>']
        real_len = min(len(essay), padding_len - 1)
        for i, one in enumerate(essay[:padding_len]):
            if one not in self.essay2idx:
                temp[i+1] = self.essay2idx['<unk>']
            else:
                temp[i+1] = self.essay2idx[one]
        essay_input = temp[:-1]
        essay_target = temp[1:]
        essay_target[real_len] = self.essay2idx['<eos>']
        if ret_tensor:
            return torch.tensor(essay_input, dtype=torch.int64), torch.tensor(essay_target, dtype=torch.int64), \
                   torch.tensor(real_len, dtype=torch.int64)
        return essay_input, essay_target, real_len

    def convert_topic2idx(self, topic, padding_num=None, ret_tensor=False):
        if padding_num == None:
            padding_num = self.topic_padding_num
        temp = [self.topic2idx['<pad_topic>'] for _ in range(padding_num)]
        real_num = min(len(topic), padding_num)
        for i, one in enumerate(topic[:padding_num]):
            if one not in self.topic2idx:
                temp[i] = self.topic2idx['<unk_topic>']
            else:
                temp[i] = self.topic2idx[one]
        if ret_tensor:
            return torch.tensor(temp, dtype=torch.int64), torch.tensor(real_num, dtype=torch.int64)
        return temp, real_num

    def print_info(self):
        tools_get_logger('data').info(
            f"data_num {len(self)} topic_num/topic_num_limit {len(self.topic2idx)}/{self.topic_num_limit} \n"
            f"essay_vocab/essay_vocab_limit {len(self.essay2idx)}/{self.essay_vocab_size} \n"
            f"topic_pad_num {self.topic_padding_num} essay_padding_len {self.essay_padding_len} \n"
            f"delete_num {len(self.delete_indexs)}"
        )

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









