import torch
from collections import defaultdict
from data import ZHIHU_dataset
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tools import tools_parse_log_file
import os
import re

class MetricDiscriminator():
    def __init__(self, fake_idx):
        self.acc_all = []
        self.acc_fake = []
        self.acc_real = []
        self.loss = []
        self.fake_idx = fake_idx

    def __call__(self, logits, label, loss):
        self.loss.append(loss)
        logits = (torch.sigmoid(logits) > 0.5)
        label = (label > 0.5)
        res = (logits == label).float()
        acc = torch.mean(res).item()
        self.acc_all.append(acc)
        real_idx = label[:, self.fake_idx] == False
        fake_idx = label[:, self.fake_idx] == True
        real_label = label[real_idx, :]
        fake_label = label[fake_idx, :]
        real_logits = logits[real_idx, :]
        fake_logits = logits[fake_idx, :]
        self.acc_fake.append(torch.mean((fake_logits == fake_label).float()).item())
        self.acc_real.append(torch.mean((real_logits == real_label).float()).item())

    def value(self):
        num = len(self.acc_all)
        return sum(self.acc_all) / num, sum(self.acc_fake) / num, sum(self.acc_real) / num, sum(self.loss) / num

    def reset(self):
        self.acc_all.clear()
        self.acc_fake.clear()
        self.acc_real.clear()
        self.loss.clear()


class MetricGenerator():
    def __init__(self, novelty_threshold=0.5):
        self.refers_train = None
        self.refers_val = None
        self.refers_test = None
        self.sm = SmoothingFunction()
        self.novelty_built_dict = {}
        self.novelty_threshold = novelty_threshold
        pass

    def jaccard_similarity(self, x0, x1):
        x0, x1 = set(x0), set(x1)
        return len(x0.intersection(x1)) / len(x0.union(x1))

    def build_novelty_dict(self, test_topics, train_topics, train_target_essays):
        for testt in test_topics:
            testt = tuple(testt)
            if testt in self.novelty_built_dict: continue
            self.novelty_built_dict[testt] = []
            temp = self.novelty_threshold
            while len(self.novelty_built_dict[testt]) == 0:
                for traint, traine in zip(train_topics, train_target_essays):
                    if self.jaccard_similarity(testt, traint) > temp:
                        self.novelty_built_dict[testt].append(traine)
                temp *= 0.5 # decrease the threshold for filling

    def novelty_evaluate(self, test_topic, generated_essay):
        max_similarity = 0.0
        test_topic = tuple(test_topic)
        for te in self.novelty_built_dict[test_topic]:
            max_similarity = max(max_similarity, self.jaccard_similarity(generated_essay, te))
        return 1 - max_similarity

    def diversity_evaluate(self, seq, official=True):
        """
        :param seq: [[0, 1, 23, ], [3, 24, 6]]
        :return:
        """
        gram2 = []
        gram1 = []
        if official:
            for one in seq:
                if isinstance(one, str): one = one.strip('\n').strip().split(' ')
                one = list(map(str, one))
                one_len = len(one)
                gram1.extend(one)
                temp2 = [f'{one[i]}_{one[i+1]}' for i in range(one_len-1)]
                gram2.extend(temp2)
            div1, div2 = len(set(gram1)) / len(gram1), len(set(gram2)) / len(gram2)
        else:
            for one in seq:
                if isinstance(one, str): one = one.strip('\n').strip().split(' ')
                one = list(map(str, one))
                one_len = len(one)
                gram1.append(len(set(one)) / one_len)
                temp2 = [f'{one[i]}_{one[i+1]}' for i in range(0, one_len-1)]
                gram2.append(len(set(temp2)) / one_len)

            div1, div2 = sum(gram1) / len(gram1), sum(gram2) / len(gram2)
        return div1, div2

    def value(self, generate_samples_idx, test_dataset:ZHIHU_dataset, train_dataset:ZHIHU_dataset=None, dataset_type='test'):
        refer_samples = []
        source_list = []
        generate_samples = []
        total_gram2_p = 0
        total_gram3_p = 0
        total_gram4_p = 0
        total_bleu2 = 0
        total_bleu3 = 0
        total_bleu4 = 0


        for i, gs in enumerate(generate_samples_idx):
            if isinstance(gs, torch.Tensor):
                generate_samples_idx[i] = gs.tolist()
            generate_samples.append(test_dataset.unpadded_idxs(generate_samples_idx[i], end_token='<eos>'))

        for si in test_dataset.data_topics:
            if isinstance(si, torch.Tensor):
                si = si.tolist()
            source_list.append(test_dataset.unpadded_idxs(si, end_token='<pad>'))


        train_target_essays = []
        train_topics = []
        if len(self.novelty_built_dict) == 0:
            assert isinstance(train_dataset, ZHIHU_dataset)
            for traint, tar in zip(train_dataset.data_topics, train_dataset.data_essays['target']):
                train_target_essays.append(train_dataset.unpadded_idxs(tar.tolist(), end_token='<eos>'))
                train_topics.append(train_dataset.unpadded_idxs(traint.tolist(), end_token='<eos>'))
            self.build_novelty_dict(source_list, train_topics, train_target_essays)


        sw = [test_dataset.convert_idx2word(sorted(x), sep='') for x in source_list]
        # sw = list(map(lambda x: "".join([idx2word[w.item()] for w in x]), sp))
        if dataset_type == 'train':
            if self.refers_train is None:
                for i, ti in enumerate(test_dataset.data_essays['target']):
                    if isinstance(ti, torch.Tensor):
                        ti = ti.tolist()
                    refer_samples.append(test_dataset.unpadded_idxs(ti, end_token='<eos>'))
                multi_refers = defaultdict(list)
                for w, r in zip(sw, refer_samples):
                    multi_refers[w].append(r)
                self.refers_train = multi_refers
            self.refers = self.refers_train
        elif dataset_type == 'test':
            if self.refers_test is None:
                multi_refers = defaultdict(list)
                for i, ti in enumerate(test_dataset.data_essays['target']):
                    if isinstance(ti, torch.Tensor):
                        ti = ti.tolist()
                    refer_samples.append(test_dataset.unpadded_idxs(ti, end_token='<eos>'))
                for w, r in zip(sw, refer_samples):
                    multi_refers[w].append(r)
                self.refers_test = multi_refers
            self.refers = self.refers_test
        elif dataset_type == 'val':
            if self.refers_val is None:
                for i, ti in enumerate(test_dataset.data_essays['target']):
                    if isinstance(ti, torch.Tensor):
                        ti = ti.tolist()
                    refer_samples.append(test_dataset.unpadded_idxs(ti, end_token='<eos>'))
                multi_refers = defaultdict(list)
                for w, r in zip(sw, refer_samples):
                    multi_refers[w].append(r)
                self.refers_val = multi_refers
            self.refers = self.refers_val


        novelty_mean = 0.0
        for w, h, t in zip(sw, generate_samples, source_list):
            refers = self.refers[w]

            novelty_mean += self.novelty_evaluate(t, h)
            total_gram2_p += sentence_bleu(refers, h, weights=(0, 1, 0, 0), smoothing_function=self.sm.method1)
            total_gram3_p += sentence_bleu(refers, h, weights=(0, 0, 1, 0), smoothing_function=self.sm.method1)
            total_gram4_p += sentence_bleu(refers, h, weights=(0, 0, 0, 1), smoothing_function=self.sm.method1)
            total_bleu2 += sentence_bleu(refers, h, weights=(0.5, 0.5, 0, 0),
                                         smoothing_function=self.sm.method1)
            total_bleu3 += sentence_bleu(refers, h, weights=(1 / 3, 1 / 3, 1 / 3, 0),
                                         smoothing_function=self.sm.method1)
            total_bleu4 += sentence_bleu(refers, h, weights=(0.25, 0.25, 0.25, 0.25),
                                         smoothing_function=self.sm.method1)

        div1, div2 = self.diversity_evaluate(generate_samples)

        return total_gram2_p / len(sw), total_gram3_p / len(sw), total_gram4_p / len(sw), total_bleu2 / len(
                sw), total_bleu3 / len(sw), total_bleu4 / len(sw), novelty_mean / len(sw), div1, div2

def evaluate_diversity_res(predictions_dir):
    writer = SummaryWriter(log_dir=predictions_dir)
    metric = MetricGenerator()
    pat = "\d+"
    best_ep = -1
    best_div1 = -1
    best_div2 = -1
    max_epoch = -1
    res = []
    for path in os.listdir(predictions_dir):
        if path.startswith('epoch_'):
            epoch = int(re.findall(pat, path)[0])
            path = f"{predictions_dir}/{path}"
            _, _, generated = tools_parse_log_file(path)
            div1, div2 = metric.diversity_evaluate(generated)
            print(f'epoch {epoch} div1 {div1:.4f} div2 {div2:.4f}')
            res.append((epoch, div1, div2))
            max_epoch = max(max_epoch, epoch)
            if div1 > best_div1:
                best_ep = epoch
                best_div1 = div1
                best_div2 = div2
    res = sorted(res, key=lambda t: t[0])
    for ep, div1, div2 in res:
        writer.add_scalar('Diversity/gram1', div1, ep)
        writer.add_scalar('Diversity/gram2', div2, ep)
    print(f'the best epoch {best_ep}/{max_epoch} div1 {best_div1:.4f} div2 {best_div2:.4f}')
    writer.flush()
    writer.close()

    return best_ep, best_div1, best_div2

def evaluate_novelty_res(predictions_dir):
    from data import read_acl_origin_data
    from config import config_zhihu_dataset, config_concepnet
    metric = MetricGenerator()
    word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem), (
        test_essay, test_topic, test_mem) \
        = read_acl_origin_data()
    train_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
                                  topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                  topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                  vocab_size=config_zhihu_dataset.vocab_size,
                                  essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                  prior=None, encode_to_tensor=True,
                                  mem_corpus_path=config_concepnet.memory_corpus_path,
                                  acl_datas=(word2idx, idx2word, topic2idx, idx2topic,
                                             (train_essay, train_topic, train_mem)))
    train_dataset.print_info()
    test_dataset = ZHIHU_dataset(path=config_zhihu_dataset.test_data_path,
                                 topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                 topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                 vocab_size=config_zhihu_dataset.vocab_size,
                                 essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                 prior=None, encode_to_tensor=True,
                                 mem_corpus_path=config_concepnet.memory_corpus_path,
                                 acl_datas=(word2idx, idx2word, topic2idx, idx2topic,
                                            (test_essay, test_topic, test_mem)))
    test_dataset.print_info()
    source_list = []
    for si in test_dataset.data_topics:
        if isinstance(si, torch.Tensor):
            si = si.tolist()
        source_list.append(test_dataset.unpadded_idxs(si, end_token='<pad>'))

    if len(metric.novelty_built_dict) == 0:
        assert isinstance(train_dataset, ZHIHU_dataset)
        train_topics = []
        train_target_essays = []
        for traint, tar in zip(train_dataset.data_topics, train_dataset.data_essays['target']):
            train_target_essays.append(train_dataset.unpadded_idxs(tar.tolist(), end_token='<eos>'))
            train_topics.append(train_dataset.unpadded_idxs(traint.tolist(), end_token='<eos>'))
        metric.build_novelty_dict(source_list, train_topics, train_target_essays)

    pat = "\d+"
    res = []
    best_ep, best_nove = -1, -1
    for path in os.listdir(predictions_dir):
        if path.startswith('epoch_'):
            novelty_mean = 0.0
            epoch = int(re.findall(pat, path)[0])
            path = f"{predictions_dir}/{path}"
            _, _, generated = tools_parse_log_file(path)
            ggg = []
            assert len(source_list) == len(generated)
            for g in generated:
                _, t, _ = train_dataset.convert_word2idx(g.split(' '))
                t = train_dataset.unpadded_idxs(t)
                ggg.append(t)
            for t, g in zip(source_list, ggg):
                novelty_mean += metric.novelty_evaluate(t, g)
            novelty_mean /= len(ggg)
            if novelty_mean > best_nove:
                best_ep = epoch
                best_nove = novelty_mean
            res.append((epoch, novelty_mean))
            print(res[-1], f'now best_epoch {best_ep} best_novelty {best_nove:.4f}')

    writer = SummaryWriter(log_dir=predictions_dir)
    res = sorted(res, key=lambda t: t[0])
    for ep, novelty in res:
        writer.add_scalar('Novelty', novelty, ep)

    return res

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath', type=str, default=None)
    parser.add_argument('--metric', type=str, help='choose [novelty | diversity]', default=None)
    args = parser.parse_args()
    assert args.dirpath
    assert args.metric
    if args.metric == 'diversity':
        evaluate_diversity_res(args.dirpath)
    elif args.metric == 'novelty':
        evaluate_novelty_res(args.dirpath)
    else:
        raise NotImplementedError(f"{args.metric} {args.dirpath} not supported")



    pass




