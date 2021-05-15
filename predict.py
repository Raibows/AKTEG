import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import random
from data import ZHIHU_dataset, read_acl_origin_data
from model_builder import build_model, activate_dropout_in_train_mode
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_make_dir, tools_to_gpu, tools_batch_idx2words, tools_write_log_to_file
from config import config_zhihu_dataset, config_train_generator, config_train_public, config_concepnet
from metric import MetricGenerator
import argparse




@torch.no_grad()
def prediction(seq2seq, train_all_dataset, test_dataset, device, res_path):
    predicts_set = []
    topics_set = []
    original_essays_set = []
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    metric = MetricGenerator()


    for i, (topic, topic_len, mems, essay_input, essay_target, essay_len) in enumerate(test_dataloader):
        topic, topic_len, mems, essay_input, essay_target, essay_len = \
            tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, essay_len, device=device)

        logits = seq2seq.forward(topic, topic_len, essay_input, essay_len+1, mems, teacher_force_ratio=False)
        # [batch, essay_len, vocab_size]
        predicts = logits.argmax(dim=-1)
        predicts_set.extend(predicts.tolist())
        topics_set.extend(topic.tolist())
        original_essays_set.extend(essay_target.tolist())

    if res_path:
        tools_make_dir(res_path)
        with open(res_path, 'w', encoding='utf-8') as file:
            for t, o, p in zip(topics_set, original_essays_set, predicts_set):
                t = test_dataset.convert_idx2word(t, sep=' ', end_token='<eos>')
                o = test_dataset.convert_idx2word(o, sep=' ', end_token='<eos>')
                p = test_dataset.convert_idx2word(p, sep=' ', end_token='<eos>')

                file.write(f'{t}\n{o}\n{p}\n')
                file.write('-*-' * int(test_dataset.essay_padding_len / 3.2))
                file.write('\n')

    gram2, gram3, gram4, bleu2, bleu3, bleu4, novelty, div1, div2 = \
        metric.value(predicts_set, test_all_dataset, train_all_dataset, dataset_type='test')
    evaluate_print = f'bleu2 {gram2:.4f} bleu3 {gram3:.4f} bleu4 {gram4:.4f}\n' \
                     f'novelty {novelty:.4f} div1 {div1:.4f} div2 {div2:.4f}\n' \
                     f'mixbleu2 {bleu2:.4f} mixbleu3 {bleu3:.4f} mixbleu4 {bleu4:.4f}\n'
    with open(res_path, 'a', encoding='utf-8') as file:
        file.write(evaluate_print)
    print(evaluate_print)
    return predicts_set



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model type', default=None)
    parser.add_argument('--load', help='model load path', default=None)
    parser.add_argument('--result', help='the result path', default=f'{tools_get_time()}.result')
    parser.add_argument('--dataset', type=str, help='chosse from [origin | acl]', default='origin')
    parser.add_argument('--device', type=str, help='choose device name like cuda:0, 1, 2...',
                        default=config_train_public.device_name)
    args = parser.parse_args()
    assert args.model and args.result and args.device and args.dataset and args.load
    tools_setup_seed(667)
    device = torch.device(args.device)
    if args.dataset == 'origin':
        train_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
                                          topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                          topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                          vocab_size=config_zhihu_dataset.vocab_size,
                                          essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                          prior=None, encode_to_tensor=True,
                                          mem_corpus_path=config_concepnet.memory_corpus_path)
        train_all_dataset.print_info()
        test_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.test_data_path,
                                         topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                         topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                         vocab_size=config_zhihu_dataset.vocab_size,
                                         essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                         mem_corpus_path=config_concepnet.memory_corpus_path,
                                         prior=train_all_dataset.get_prior(), encode_to_tensor=True)
        test_all_dataset.print_info()
    elif args.dataset == 'acl':
        word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem), (
            test_essay, test_topic, test_mem) \
            = read_acl_origin_data()
        train_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
                                          topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                          topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                          vocab_size=config_zhihu_dataset.vocab_size,
                                          essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                          prior=None, encode_to_tensor=True,
                                          mem_corpus_path=config_concepnet.memory_corpus_path,
                                          acl_datas=(word2idx, idx2word, topic2idx, idx2topic,
                                                     (train_essay, train_topic, train_mem)))
        train_all_dataset.print_info()
        test_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.test_data_path,
                                         topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                         topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                         vocab_size=config_zhihu_dataset.vocab_size,
                                         essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                         prior=None, encode_to_tensor=True,
                                         mem_corpus_path=config_concepnet.memory_corpus_path,
                                         acl_datas=(word2idx, idx2word, topic2idx, idx2topic,
                                                    (test_essay, test_topic, test_mem)))
        test_all_dataset.print_info()
    else:
        raise NotImplementedError(f'{args.dataset} not supported')

    seq2seq = build_model(model_name=args.model, dataset_name=args.dataset,
                          vocab_size=len(train_all_dataset.word2idx), device=device,
                          load_path=args.load, init_way='normal',
                          mask_idx=train_all_dataset.word2idx['<pad>'])
    seq2seq.eval()
    seq2seq = activate_dropout_in_train_mode(seq2seq)
    prediction(seq2seq, train_all_dataset, test_all_dataset, device, res_path=args.result)