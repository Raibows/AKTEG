import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import ZHIHU_dataset
from neural import KnowledgeEnhancedSeq2Seq
from tools import tools_get_logger, tools_setup_seed, tools_make_dir, tools_to_gpu, tools_get_time
from config import config_zhihu_dataset, config_train_generator, config_seq2seq, config_concepnet, config_train_public
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model load path', default=config_seq2seq.model_load_path)
parser.add_argument('--result', help='the result path', default=f'{tools_get_time()}.result')
args = parser.parse_args()
assert args.model and args.result


@torch.no_grad()
def prediction(seq2seq, train_all_dataset, dataset_loader, device, res_path):
    seq2seq.eval()
    teacher_force_ratio = 0.0
    predicts_set = []
    topics_set = []
    original_essays_set = []
    for i, (topic, topic_len, mems, essay_input, essay_target, _) in enumerate(dataset_loader):
        topic, topic_len, mems, essay_input, essay_target = \
            tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, device=device)

        logits = seq2seq.forward(topic, topic_len, essay_input, mems, teacher_force_ratio=teacher_force_ratio)
        # [batch, essay_len, vocab_size]
        predicts = logits.argmax(dim=-1)
        predicts_set.extend(predicts.tolist())
        topics_set.extend(topic.tolist())
        original_essays_set.extend(essay_target.tolist())


    tools_make_dir(res_path)
    with open(res_path, 'w', encoding='utf-8') as file:
        for t, o, p in zip(topics_set, original_essays_set, predicts_set):
            t = train_all_dataset.convert_idx2word(t, sep=' ')
            o = train_all_dataset.convert_idx2word(o, sep='')
            p = train_all_dataset.convert_idx2word(p, sep='')

            file.write(f'{t}\n{o}\n{p}\n')
            file.write('--' * train_all_dataset.essay_padding_len)
            file.write('\n')
        file.write(f'the evaluate model is {args.model} on {config_zhihu_dataset.test_data_path} in {tools_get_time()}')

    tools_get_logger('predict').info(f'model {args.model} predicts '
                                     f'{config_zhihu_dataset.test_data_path} results to {res_path}')



if __name__ == '__main__':
    tools_setup_seed(667)
    device = torch.device('cuda:3')
    tools_get_logger('predict').info('using cuda : 3')

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

    test_all_dataloader = DataLoader(test_all_dataset, batch_size=config_train_generator.batch_size,
                                     num_workers=config_train_public.dataloader_num_workers, pin_memory=True)

    tools_get_logger('train').info(f"load train data {len(train_all_dataset)} test data {len(test_all_dataset)} "
                                   f"test/all {len(test_all_dataset) / len(train_all_dataset):.4f}")

    seq2seq = KnowledgeEnhancedSeq2Seq(vocab_size=len(train_all_dataset.word2idx),
                                       embed_size=config_seq2seq.embedding_size,
                                       pretrained_wv_path=config_seq2seq.pretrained_wv_path['tencent'],
                                       encoder_lstm_hidden=config_seq2seq.encoder_lstm_hidden_size,
                                       encoder_bid=config_seq2seq.encoder_lstm_is_bid,
                                       lstm_layer=config_seq2seq.lstm_layer_num,
                                       attention_size=config_seq2seq.attention_size,
                                       device=device)
    tools_get_logger('predict').info(f"loading model from {args.model}")
    seq2seq.load_state_dict(torch.load(args.model, map_location=device))
    seq2seq.to(device)
    seq2seq.eval()

    prediction(seq2seq, train_all_dataset, test_all_dataloader, device, res_path=args.result)