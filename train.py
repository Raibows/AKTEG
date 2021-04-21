import torch
import numpy as np
import random
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import ZHIHU_dataset
from neural import Encoder, Decoder, Seq2Seq
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_make_dir, tools_copy_file
from preprocess import k_fold_split
from config import config_zhihu_dataset, config_train, config_seq2seq


tools_setup_seed(667)

device = torch.device(config_zhihu_dataset.device_name)

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
                                        'idx2essay': train_all_dataset.idx2essay})

test_all_dataloader = DataLoader(test_all_dataset, batch_size=config_train.batch_size)

tools_get_logger('data').info(f"load train data {len(train_all_dataset)} test data {len(test_all_dataset)}")

encoder = Encoder(vocab_size=train_all_dataset.topic_num_limit,
                  embed_size=config_seq2seq.encoder_embed_size,
                  layer_num=config_seq2seq.encoder_lstm_layer_num,
                  hidden_size=config_seq2seq.encoder_lstm_hidden_size,
                  is_bid=config_seq2seq.encoder_lstm_is_bid,
                  pretrained_path=config_zhihu_dataset.topic_preprocess_wv_path)

decoder = Decoder(vocab_size=train_all_dataset.essay_vocab_size,
                  embed_size=config_seq2seq.decoder_embed_size,
                  layer_num=config_seq2seq.decoder_lstm_layer_num,
                  hidden_size=encoder.output_size,
                  pretrained_path=config_zhihu_dataset.essay_preprocess_wv_path)

seq2seq = Seq2Seq(encoder, decoder, train_all_dataset.essay_vocab_size, device)
seq2seq.to(device)

optimizer = optim.AdamW(seq2seq.parameters(), lr=config_train.learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=train_all_dataset.essay2idx['<pad>']).to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=3, min_lr=6e-6)
warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1e-2 if ep < 3 else 1)

def to_gpu(*params, device=torch.device('cpu')):
    return [p.to(device) for p in params]


def train(train_all_dataset, dataset_loader):
    seq2seq.train()
    loss_mean = 0.0
    teacher_force_ratio = config_seq2seq.teacher_force_rate
    with tqdm(total=len(dataset_loader), desc='train') as pbar:
        for topic, topic_len, essay_input, essay_target, _ in dataset_loader:

            topic, topic_len, essay_input, essay_target = to_gpu(topic, topic_len, essay_input, essay_target, device=device)

            logits = seq2seq.forward((topic, topic_len), essay_input, teacher_force_ratio=teacher_force_ratio)

            essay_target = essay_target.view(-1)
            logits = logits.view(-1, train_all_dataset.essay_vocab_size)
            optimizer.zero_grad()
            loss = criterion(logits, essay_target)
            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=1.0, norm_type=2.0)
            optimizer.step()

            pbar.set_postfix_str(f"loss: {loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)
    return loss_mean / len(dataset_loader)

@torch.no_grad()
def validation(train_all_dataset, dataset_loader):
    seq2seq.eval()
    loss_mean = 0.0
    teacher_force_ratio = 0.0
    optimizer.zero_grad()
    with tqdm(total=len(dataset_loader), desc='validation') as pbar:
        for topic, topic_len, essay_input, essay_target, _ in dataset_loader:

            topic, topic_len, essay_input, essay_target = to_gpu(topic, topic_len, essay_input, essay_target,
                                                                 device=device)

            logits = seq2seq.forward((topic, topic_len), essay_input, teacher_force_ratio=teacher_force_ratio)

            essay_target = essay_target.view(-1)
            logits = logits.view(-1, train_all_dataset.essay_vocab_size)
            loss = criterion(logits, essay_target)

            pbar.set_postfix_str(f"loss: {loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)
    return loss_mean / len(dataset_loader)




if __name__ == '__main__':
    writer = tools_get_tensorboard_writer()
    best_save_loss = 1e9
    for ep in range(config_train.epoch):
        kfolds = k_fold_split(train_all_dataset, config_train.batch_size, k=config_train.fold_k)
        train_loss = 0.0
        valid_loss = 0.0
        for fold_no, (train_dataloader, valid_dataloader) in enumerate(kfolds):
            train_loss_t = train(train_all_dataset, train_dataloader)
            valid_loss_t = validation(train_all_dataset, valid_dataloader)
            train_loss += train_loss_t
            valid_loss += valid_loss_t
            tools_get_logger('train').info(f'epoch {ep} fold {fold_no} done '
                                           f'train_loss {train_loss_t:.4f} valid_loss {valid_loss_t:.4f}')

        test_loss = validation(train_all_dataset, test_all_dataloader)
        scheduler.step(test_loss)
        warmup_scheduler.step(ep)

        train_loss /= len(kfolds)
        valid_loss /= len(kfolds)
        del kfolds
        writer.add_scalar('Loss/train', train_loss, ep)
        writer.add_scalar('Loss/valid', valid_loss, ep)
        writer.add_scalar('Loss/test', test_loss, ep)

        tools_get_logger('train').info(f'epoch {ep} done train_loss {train_loss:.4f} '
                                       f'valid_loss {valid_loss:.4f} test_loss {test_loss:.4f}')

        if test_loss < best_save_loss:
            save_path = config_seq2seq.model_save_fmt.format(tools_get_time(), test_loss)
            tools_make_dir(save_path)
            tools_copy_file('./config.py', save_path+'.config.py')
            torch.save(seq2seq.state_dict(), save_path)
            best_save_loss = test_loss
            tools_get_logger('train').info(f"saving model to {save_path}, now best_test_loss {best_save_loss:.4f}")








    writer.flush()
    writer.close()