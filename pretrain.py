import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse
import os
from data import ZHIHU_dataset, read_acl_origin_data
from neural import KnowledgeEnhancedSeq2Seq, simple_seq2seq, init_param
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_make_dir, tools_copy_all_suffix_files, tools_to_gpu, \
    tools_check_if_in_debug_mode, tools_write_log_to_file
from transformer import KnowledgeTransformerSeq2Seqv3
from magic import MagicSeq2Seq
from config import config_zhihu_dataset, config_train_generator, config_seq2seq, config_train_public, config_concepnet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='choose [simple|knowledge|attention|magic]', default='knowledge')
parser.add_argument('--device', type=str, help='choose device name like cuda:0, 1, 2...', default=config_train_public.device_name)
parser.add_argument('--epoch', type=int, help='epoch num default is config_epoch', const=config_train_generator.epoch, nargs='?')
parser.add_argument('--batch', type=int, help='batch size default is config_batch', const=config_train_generator.batch_size, nargs='?')
args = parser.parse_args()
if not args.device.startswith('cuda:') and args.device != 'cpu':
    args.device = config_train_public.device_name
args.dataset = 'acl'

if tools_check_if_in_debug_mode():
    args.device = 'cpu'
    args.epoch = 2
    args.batch = 4
    config_train_public.dataloader_num_workers = 0

tools_get_logger('pretrain').info(f"pid {os.getpid()} using device {args.device} training on acl with model {args.model} epoch {args.epoch} batch {args.batch}")

tools_setup_seed(667)
device = torch.device(args.device)


def pretrain_process(train_all_dataset, seq2seq, writer_logdir_starttime, epoch_num, batch_size):
    seq2seq.train()
    dataloader = DataLoader(train_all_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = optim.AdamW(seq2seq.parameters(), lr=config_train_generator.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=train_all_dataset.word2idx['<pad>']).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=5, min_lr=9e-5)
    writer, log_dir, start_time = writer_logdir_starttime
    print_interval = len(dataloader) // 5
    begin_teacher_force_ratio = 1.0
    tools_copy_all_suffix_files(target_dir=f'{log_dir}/pyfile/', source_dir='.', suffix='.py')

    best_save_loss = 1e10
    best_save_path = None

    for ep in range(epoch_num):
        if ep > 60:
            begin_teacher_force_ratio *= 0.95
            begin_teacher_force_ratio = max(begin_teacher_force_ratio, 0.75)
        loss_mean = 0.0
        with tqdm(total=len(dataloader), desc=f'pretrain_{ep}') as pbar:
            for i, (_, _, mems, essay_input, essay_target, essay_len) in enumerate(dataloader):
                mems, essay_input, essay_target, essay_len = \
                    tools_to_gpu(mems, essay_input, essay_target, essay_len, device=device)
                logits = seq2seq.forward(essay_input[:, 1:], essay_len, essay_input, essay_len+1, mems,
                                         teacher_force_ratio=begin_teacher_force_ratio)
                optimizer.zero_grad()
                loss = criterion(logits.view(-1, len(train_all_dataset.word2idx)), essay_target.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=config_train_generator.grad_clip_max_norm,
                                         norm_type=config_train_generator.grad_clip_norm_type)
                optimizer.step()
                loss_mean += loss.item()
                pbar.set_postfix_str(f"loss: {loss.item():.4f}")
                pbar.update(1)
                if (i+1) % print_interval == 0:
                    sample = logits[random.randint(0, logits.shape[0] - 1)]
                    idx = sample.argmax(dim=-1)
                    idx = idx.squeeze()[:10].tolist()
                    tools_get_logger('example').info(f"{train_all_dataset.convert_idx2word(idx)}\n")

        loss_mean /= len(dataloader)
        scheduler.step(loss_mean, epoch=ep)
        writer.add_scalar(f'pretrain_seq2seq_{args.model}/train_loss', loss_mean, ep)

        if loss_mean < best_save_loss:
            best_save_loss = loss_mean
            save_dir = f"{log_dir}/model_state/"
            tools_make_dir(save_dir)
            best_save_path = f"{save_dir}epoch_{ep}_loss_{loss_mean:.4f}.pt"
            torch.save(seq2seq.state_dict(), best_save_path)

        tools_get_logger('pretrain').info(f'epoch {ep} done! train_loss {loss_mean} now_best_model {best_save_path}')

    tools_get_logger('pretrain').info(f'{epoch_num} done ! best model is in {best_save_path}')





if __name__ == '__main__':

    word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem), (
    test_essay, test_topic, test_mem) \
        = read_acl_origin_data()
    train_essay += test_essay
    train_topic += test_topic
    train_mem = np.concatenate((train_mem, test_mem), axis=0)
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

    if args.model == 'knowledge':
        seq2seq = KnowledgeEnhancedSeq2Seq(vocab_size=len(train_all_dataset.word2idx),
                                           embed_size=config_seq2seq.embedding_size,
                                           pretrained_wv_path=config_seq2seq.pretrained_wv_path[args.dataset],
                                           encoder_lstm_hidden=config_seq2seq.encoder_lstm_hidden_size,
                                           encoder_bid=config_seq2seq.encoder_lstm_is_bid,
                                           lstm_layer=config_seq2seq.lstm_layer_num,
                                           attention_size=config_seq2seq.attention_size,
                                           device=device)
    elif args.model == 'simple':
        seq2seq = simple_seq2seq(2, 128, len(train_all_dataset.word2idx), 128, device)
    elif args.model == 'transformer':
        seq2seq = KnowledgeTransformerSeq2Seqv3(vocab_size=len(train_all_dataset.word2idx),
                                     embed_size=config_seq2seq.embedding_size,
                                     pretrained_wv_path=config_seq2seq.pretrained_wv_path[args.dataset],
                                     device=device,
                                     mask_idx=train_all_dataset.word2idx['<pad>'])
    elif args.model == 'magic':
        seq2seq = MagicSeq2Seq(vocab_size=len(train_all_dataset.word2idx),
                               embed_size=config_seq2seq.embedding_size,
                               pretrained_wv_path=config_seq2seq.pretrained_wv_path[args.dataset],
                               encoder_lstm_hidden=512,
                               encoder_bid=True,
                               lstm_layer=1,
                               device=device)
    else:
        raise NotImplementedError(f'{args.model} not supported')

    init_param(seq2seq, init_way=config_train_generator.model_init_way)

    if config_seq2seq.model_load_path:
        seq2seq.load_state_dict(torch.load(config_seq2seq.model_load_path, map_location=device))
    seq2seq.to(device)
    seq2seq.eval()

    writer, log_dir, start_time = tools_get_tensorboard_writer(dir_pre=f'pretrain_seq2seq_{args.model}')

    pretrain_process(epoch_num=args.epoch,
                     batch_size=args.batch,
                     writer_logdir_starttime=(writer, log_dir, start_time),
                     train_all_dataset=train_all_dataset,
                     seq2seq=seq2seq)
    writer.flush()
    writer.close()
