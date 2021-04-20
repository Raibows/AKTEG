import copy

import torch
from torch.utils.data import Dataset


class ZHIHU_dataset(Dataset):
    def __init__(self, path, topic_num_limit, essay_vocab_size, topic_threshold, topic_padding_num, essay_padding_len,
                 topic_special_tokens={'<pad_topic>': 0, '<unk_topic>': 1, '<fake_topic>': 2},
                 essay_special_tokens={'<sos>': 0, '<eos>': 1, '<unk>': 2, '<pad>': 3}, ):

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

        temp_topic2idx, temp_essay2idx = self.__read_datas(essay_special_tokens, topic_special_tokens)
        self.topic2idx, self.idx2topic = self.__limit_size(topic_special_tokens, temp_topic2idx, topic_num_limit, self.data_topics)
        self.essay2idx, self.idx2essay = self.__limit_size(essay_special_tokens, temp_essay2idx, essay_vocab_size, self.data_essays)
        self.__limit_datas()
        self.__encode_datas()


    def __encode_datas(self):
        assert len(self.data_topics) == len(self.data_essays)
        self.len_topics = [0 for _ in self.data_topics]
        self.len_essays = [0 for _ in self.data_essays]
        for i, (t, e) in enumerate(zip(self.data_topics, self.data_essays)):
            self.data_topics[i], self.len_topics[i] = self.convert_topic2idx(t, ret_tensor=True)
            self.data_essays[i], self.len_essays[i] = self.convert_essay2idx(e, ret_tensor=True)

    def __limit_size(self, reserved, temp2idx, size, datas):
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

    def __limit_datas(self):
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

    def __preprocess(self, sent):
        temp = []
        for x in sent.split(' '):
            if x and (not x.isspace()):
                temp.append(x.strip())
        return temp

    def __read_datas(self, essay2idx, topic2idx):
        topics = []
        labels = []
        essay2idx = copy.deepcopy(essay2idx)
        topic2idx = copy.deepcopy(topic2idx)
        with open(self.path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n').strip('\r').split('</d>')
                # process the essay
                temp = self.__preprocess(line[0])
                for l in temp:
                    if l not in essay2idx: essay2idx[l] = len(essay2idx)
                labels.append(temp)
                # process the topics
                temp = self.__preprocess(line[1])
                for t in temp:
                    if t not in topic2idx: topic2idx[t] = len(topic2idx)
                topics.append(temp)

        self.data_topics = topics
        self.data_essays = labels

        return topic2idx, essay2idx

    def convert_idx2essay(self, idxs):
        return [self.idx2essay[i] for i in idxs]

    def convert_idx2topic(self, idxs):
        return [self.idx2topic[i] for i in idxs]

    def convert_essay2idx(self, essay, padding_len=None, ret_tensor=False):
        if padding_len == None:
            padding_len = self.essay_padding_len
        temp = [self.essay2idx['<pad>'] for _ in range(padding_len)]
        real_len = min(len(essay), padding_len)
        for i, one in enumerate(essay[:padding_len]):
            if one not in self.essay2idx:
                temp[i] = self.essay2idx['<unk>']
            else:
                temp[i] = self.essay2idx[one]
        if ret_tensor:
            return torch.tensor(temp, dtype=torch.int32), torch.tensor(real_len, dtype=torch.int32)
        return temp, real_len

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
            return torch.tensor(temp, dtype=torch.int32), torch.tensor(real_num, dtype=torch.int32)
        return temp, real_num

    def __len__(self):
        return len(self.data_topics)

    def __getitem__(self, item):
        return (self.data_topics[item], self.len_topics[item], self.data_essays[item], self.len_essays[item])



if __name__ == '__main__':
    train_dataset = ZHIHU_dataset('../data/zhihu.txt', 103, 50004, 4, 5, 100)
















