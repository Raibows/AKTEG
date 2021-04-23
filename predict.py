import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import ZHIHU_dataset
from neural import Encoder, Decoder, Seq2Seq, Memory_neural
from tools import tools_get_logger, tools_setup_seed, tools_make_dir, tools_to_gpu
from config import config_zhihu_dataset, config_train, config_seq2seq, config_concepnet



@torch.no_grad()
def prediction(seq2seq, train_all_dataset, dataset_loader, device, res_path=config_seq2seq.model_load_path):
    seq2seq.eval()
    teacher_force_ratio = 0.0
    predicts_set = []
    topics_set = []
    original_essays_set = []
    with tqdm(total=len(dataset_loader), desc='prediction') as pbar:
        for topic, topic_len, mems, essay_input, essay_target, _ in dataset_loader:

            topic, topic_len, mems, essay_input, essay_target = \
                tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, device=device)

            logits = seq2seq.forward((topic, topic_len), essay_input, mems=mems, teacher_force_ratio=teacher_force_ratio)
            logits = logits.permute(1, 0, 2) # [batch, essay_max_len, essay_vocab_size]
            # print(logits.shape)
            # print(logits[0][0])
            # temp = logits[0][0].argmax().item()
            # print(temp, train_all_dataset.convert_idx2essay([temp]))
            # break
            predicts = logits.argmax(dim=2)
            predicts_set.extend(predicts.tolist())
            topics_set.extend(topic.tolist())
            original_essays_set.extend(essay_target.tolist())

            pbar.update(1)

    res_path = f'{res_path}.predictions.load_model_{config_train.is_load_model}'
    tools_make_dir(res_path)
    with open(res_path, 'w', encoding='utf-8') as file:
        for t, o, p in zip(topics_set, original_essays_set, predicts_set):
            t, o, p = train_all_dataset.convert_idx2topic(t), train_all_dataset.convert_idx2essay(o), \
            train_all_dataset.convert_idx2essay(p)
            file.write(' '.join(t))
            file.write('\n')
            file.write(' '.join(o))
            file.write('\n')
            file.write(' '.join(p))
            file.write('\n')

    tools_get_logger('prediction').info(f"predictions done! the res is in {res_path}")



if __name__ == '__main__':
    tools_setup_seed(667)

    device = torch.device(config_train.device_name)

    train_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
                                      topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                      essay_vocab_size=config_zhihu_dataset.essay_vocab_size,
                                      topic_threshold=config_zhihu_dataset.topic_threshold,
                                      topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                      essay_padding_len=config_zhihu_dataset.essay_padding_len)

    test_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.test_data_path,
                                     topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                     essay_vocab_size=config_zhihu_dataset.essay_vocab_size,
                                     topic_threshold=config_zhihu_dataset.topic_threshold,
                                     topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                     essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                     prior={'topic2idx': train_all_dataset.topic2idx,
                                            'idx2topic': train_all_dataset.idx2topic,
                                            'essay2idx': train_all_dataset.essay2idx,
                                            'idx2essay': train_all_dataset.idx2essay,
                                            'mem2idx': train_all_dataset.mem2idx,
                                            'idx2mem': train_all_dataset.idx2mem,
                                            'memory_corpus': train_all_dataset.memory_corpus})

    test_all_dataloader = DataLoader(test_all_dataset, batch_size=config_train.batch_size * 2,
                                     num_workers=config_train.dataloader_num_workers, pin_memory=True)

    tools_get_logger('train').info(f"load train data {len(train_all_dataset)} test data {len(test_all_dataset)}")

    encoder = Encoder(vocab_size=train_all_dataset.topic_num_limit,
                      embed_size=config_seq2seq.encoder_embed_size,
                      layer_num=config_seq2seq.encoder_lstm_layer_num,
                      hidden_size=config_seq2seq.encoder_lstm_hidden_size,
                      is_bid=config_seq2seq.encoder_lstm_is_bid,
                      pretrained_path=config_zhihu_dataset.topic_preprocess_wv_path)

    decoder = Decoder(vocab_size=train_all_dataset.essay_vocab_size,
                      embed_size=config_seq2seq.decoder_embed_size,
                      layer_num=config_seq2seq.decoder_lstm_layer_num,
                      encoder_output_size=encoder.output_size,
                      memory_neural_embed_size=config_seq2seq.memory_embed_size,
                      pretrained_path=config_zhihu_dataset.essay_preprocess_wv_path)

    memory_neural = Memory_neural(vocab_size=train_all_dataset.mem_vocab_size,
                                  embed_size=config_seq2seq.memory_embed_size,
                                  decoder_hidden_size=decoder.hidden_size,
                                  decoder_embed_size=decoder.embed_size,
                                  pretrained_path=config_concepnet.memory_pretrained_wv_path)

    seq2seq = Seq2Seq(encoder=encoder,
                      decoder=decoder,
                      memory_neural=memory_neural,
                      topic_padding_num=train_all_dataset.topic_padding_num,
                      essay_vocab_size=train_all_dataset.essay_vocab_size,
                      attention_size=config_seq2seq.attention_size,
                      device=device)
    if config_train.is_load_model:
        tools_get_logger('prediction').info(f"loading model from {config_seq2seq.model_load_path}")
        seq2seq.load_state_dict(torch.load(config_seq2seq.model_load_path, map_location=device))
    seq2seq.to(device)
    seq2seq.eval()


    prediction(seq2seq, train_all_dataset, test_all_dataloader, device)