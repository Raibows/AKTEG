import torch
import numpy as np
import random
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import ZHIHU_dataset
from neural import Encoder, Decoder, Seq2Seq
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_k_fold_split
from config import config_zhihu_dataset, config_train


tools_setup_seed(667)

device = torch.device('cuda:0')

train_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
                                  topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                  essay_vocab_size=config_zhihu_dataset.essay_vocab_size,
                                  topic_threshold=config_zhihu_dataset.topic_threshold,
                                  topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                  essay_padding_len=config_zhihu_dataset.essay_padding_len)



encoder = Encoder(train_all_dataset.topic_num_limit, 100, layer_num=2, hidden_size=100, is_bid=True)
decoder = Decoder(train_all_dataset.essay_vocab_size, 100, 2, encoder.output_size)
seq2seq = Seq2Seq(encoder, decoder, train_all_dataset.essay_vocab_size, device)
seq2seq.to(device)

optimizer = optim.AdamW(seq2seq.parameters(), lr=config_train.learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=train_all_dataset.essay2idx['<pad>']).to(device)

def to_gpu(*params, device=torch.device('cpu')):
    return [p.to(device) for p in params]


def train(train_all_dataset, dataset_loader):
    seq2seq.train()
    loss_mean = 0.0
    teacher_force_ratio = config_train.train_teacher_force_rate
    with tqdm(total=len(dataset_loader), desc='train') as pbar:
        for topic, topic_len, essay, essay_len in dataset_loader:
            temp_sos = torch.full([essay.size(0), 1], train_all_dataset.essay2idx['<sos>'])
            temp_eos = torch.full([essay.size(0), 1], train_all_dataset.essay2idx['<eos>'])
            essay_input = torch.cat([temp_sos, essay], dim=1)
            essay_target = torch.cat([essay, temp_eos], dim=1)
            essay_len += 1

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
        for topic, topic_len, essay, essay_len in dataset_loader:
            temp_sos = torch.full([essay.size(0), 1], train_all_dataset.essay2idx['<sos>'])
            temp_eos = torch.full([essay.size(0), 1], train_all_dataset.essay2idx['<eos>'])
            essay_input = torch.cat([temp_sos, essay], dim=1)
            essay_target = torch.cat([essay, temp_eos], dim=1)
            essay_len += 1

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
    for ep in range(config_train.epoch):
        kfolds = tools_k_fold_split(train_all_dataset, config_train.batch_size, k=config_train.fold_k)
        train_loss = 0.0
        valid_loss = 0.0
        for fold_no, (train_dataloader, valid_dataloader) in enumerate(kfolds):
            train_loss_t = train(train_all_dataset, train_dataloader)
            valid_loss_t = validation(train_all_dataset, valid_dataloader)
            train_loss += train_loss_t
            valid_loss += valid_loss_t
            tools_get_logger('train').info(f'epoch {ep} fold {fold_no} done train loss {train_loss_t} valid loss {valid_loss_t}')

        train_loss /= len(kfolds)
        valid_loss /= len(kfolds)
        del kfolds
        writer.add_scalar('Loss/train', train_loss, ep)
        writer.add_scalar('Loss/valid', valid_loss, ep)








    writer.flush()
    writer.close()