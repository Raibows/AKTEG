import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import random
import argparse
import os
from data import ZHIHU_dataset, read_acl_origin_data
from model_builder import build_model, activate_dropout_in_train_mode
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_make_dir, tools_copy_all_suffix_files, tools_to_gpu, \
    tools_check_if_in_debug_mode, tools_write_log_to_file
from config import config_zhihu_dataset, config_train_generator, config_train_public, config_concepnet
from metric import MetricGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='choose [simple|paper|attention|magic|cteg]', default='cteg')
parser.add_argument('--device', type=str, help='choose device name like cuda:0, 1, 2...', default=config_train_public.device_name)
parser.add_argument('--dataset', type=str, help='chosse from [origin | acl | expand]', default='origin')
parser.add_argument('--epoch', type=int, help='epoch num default is config_epoch', const=config_train_generator.epoch, nargs='?')
parser.add_argument('--batch', type=int, help='batch size default is config_batch', const=config_train_generator.batch_size, nargs='?')
parser.add_argument('--load', type=str, help='load the pretrained model', nargs='?', const=None)
args = parser.parse_args()
if not args.device.startswith('cuda:') and args.device != 'cpu':
    args.device = config_train_public.device_name

if tools_check_if_in_debug_mode():
    args.device = 'cpu'
    args.epoch = 2
    args.batch = 3
    config_train_public.dataloader_num_workers = 0
    writer_pre = 'public'
else:
    writer_pre = args.model

tools_get_logger('train_G').info(f"pid {os.getpid()} using device {args.device} training on {args.dataset} with model {args.model} epoch {args.epoch} batch {args.batch}")

tools_setup_seed(667)
device = torch.device(args.device)

def train_generator(epoch, train_all_dataset, dataset_loader, seq2seq, optimizer, criterion, teacher_force):
    seq2seq.train()
    loss_mean = 0.0
    with tqdm(total=len(dataset_loader), desc=f'train{epoch}') as pbar:
        for i, (topic, topic_len, mems, essay_input, essay_target, essay_len) in enumerate(dataset_loader):

            topic, topic_len, mems, essay_input, essay_target, essay_len = \
                tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, essay_len, device=device)

            logits = seq2seq.forward(topic, topic_len, essay_input, essay_len+1, mems, teacher_force=teacher_force)
            # [batch, essay_max_len, essay_vocab_size]
            logits = logits.view(-1, len(train_all_dataset.word2idx))
            optimizer.zero_grad()
            loss = criterion(logits, essay_target.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=config_train_generator.grad_clip_max_norm,
                                     norm_type=config_train_generator.grad_clip_norm_type)
            optimizer.step()
            pbar.set_postfix_str(f"loss: {loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)

    return loss_mean / len(dataset_loader)

@torch.no_grad()
def test_generator(epoch, metric:MetricGenerator, test_all_dataset, dataset_loader, train_dataset,
                   seq2seq, criterion, prediction_path=None, dataset_type='test'):
    seq2seq.eval()
    seq2seq = activate_dropout_in_train_mode(seq2seq)
    loss_mean = 0.0
    teacher_force = False
    show_interval = len(dataset_loader) // 5
    predicts_set = []
    topics_set = []
    original_essays_set = []


    with tqdm(total=len(dataset_loader), desc=f'validation{epoch}') as pbar:
        for i, (topic, topic_len, mems, essay_input, essay_target, essay_len) in enumerate(dataset_loader):
            pass
            topic, topic_len, mems, essay_input, essay_target, essay_len = \
                tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, essay_len, device=device)

            logits = seq2seq.forward(topic, topic_len, essay_input, essay_len+1, mems, teacher_force=teacher_force)
            # [batch, essay_len, vocab_size]
            if (i+1) % show_interval == 0:
                sample = logits[random.randint(0, logits.shape[0]-1)]
                idx = sample.argmax(dim=-1)
                idx = idx.squeeze()[:10].tolist()
                tools_get_logger('example').info(f"\n{test_all_dataset.convert_idx2topic(topic[0].tolist())}\n"
                                                 f"{test_all_dataset.convert_idx2word(idx)}\n")

            if prediction_path:
                predicts = logits.argmax(dim=-1)
                predicts_set.extend(predicts.tolist())
                topics_set.extend(topic.tolist())
                original_essays_set.extend(essay_target.tolist())


            loss = criterion(logits.view(-1, len(test_all_dataset.word2idx)), essay_target.view(-1)).item()
            loss_mean += loss
            pbar.set_postfix_str(f"loss: {loss:.4f}")
            pbar.update(1)

    if prediction_path:
        tools_make_dir(prediction_path)
        with open(prediction_path, 'w', encoding='utf-8') as file:
            for t, o, p in zip(topics_set, original_essays_set, predicts_set):
                t = test_all_dataset.convert_idx2word(t, sep=' ', end_token='<eos>')
                o = test_all_dataset.convert_idx2word(o, sep=' ', end_token='<eos>')
                p = test_all_dataset.convert_idx2word(p, sep=' ', end_token='<eos>')

                file.write(f'{t}\n{o}\n{p}\n')
                file.write('-*-' * int(test_all_dataset.essay_padding_len / 3.2))
                file.write('\n')

        tools_get_logger('validation').info(f"predictions epoch{epoch} done! the res is in {prediction_path}")

    gram2, gram3, gram4, bleu2, bleu3, bleu4, novelty, div1, div2 = \
        metric.value(predicts_set, test_all_dataset, train_dataset, dataset_type=dataset_type)

    return loss_mean / len(dataset_loader), gram2, gram3, gram4, bleu2, bleu3, bleu4, novelty, div1, div2

def train_generator_process(epoch_num, train_all_dataset, test_all_dataset, seq2seq, batch_size, writer_logdir_starttime):
    writer, log_dir, start_time = writer_logdir_starttime

    test_all_dataloader = DataLoader(test_all_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=config_train_public.dataloader_num_workers, pin_memory=args.device != 'cpu')
    train_all_dataloader = DataLoader(train_all_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=config_train_public.dataloader_num_workers, pin_memory=args.device != 'cpu')

    tools_get_logger('train').info(f"load train data {len(train_all_dataset)} test data {len(test_all_dataset)} "
                                   f"test/train {len(test_all_dataset) / len(train_all_dataset):.4f}")

    metric = MetricGenerator()

    # from scratch
    optimizer = optim.Adam(seq2seq.parameters(), lr=config_train_generator.learning_rate)


    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=train_all_dataset.word2idx['<pad>']).to(device)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=5, min_lr=9e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_num, eta_min=1e-9, last_epoch=-1, verbose=True)
    warmup_epoch = -1
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1e-2 if ep < warmup_epoch else 1.0)

    tools_copy_all_suffix_files(target_dir=f'{log_dir}/pyfile/', source_dir='.', suffix='.py')

    for ep in range(epoch_num):
        train_loss = train_generator(ep, train_all_dataset, train_all_dataloader, seq2seq,
                                           optimizer, criterion, teacher_force=True)

        prediction_path = f'{log_dir}/epoch_{ep}.predictions'
        test_loss, gram2, gram3, gram4, bleu2, bleu3, bleu4, novelty, div1, div2 = \
        test_generator(ep, metric, test_all_dataset, test_all_dataloader, train_all_dataset,
                           seq2seq, criterion, prediction_path=prediction_path, dataset_type='test')

        if ep > warmup_epoch:
            scheduler.step()
        else:
            warmup_scheduler.step()

        writer.add_scalar('Loss/train', train_loss, ep)
        writer.add_scalar('Loss/test', test_loss, ep)
        writer.add_scalar('Bleu/gram2', gram2, ep)
        writer.add_scalar('Bleu/mixgram4', bleu4, ep)
        writer.add_scalar('Novelty', novelty, ep)
        writer.add_scalar('Diversity/gram1', div1, ep)
        writer.add_scalar('Diversity/gram2', div2, ep)

        evaluate_print = f'train_loss {train_loss:.4f} test_loss {test_loss:.4f}\n' \
                         f'bleu2 {gram2:.4f} bleu3 {gram3:.4f} bleu4 {gram4:.4f}\n' \
                         f'novelty {novelty:.4f} div1 {div1:.4f} div2 {div2:.4f}\n' \
                         f'mixbleu2 {bleu2:.4f} mixbleu3 {bleu3:.4f} mixbleu4 {bleu4:.4f}\n'
        with open(prediction_path, 'a', encoding='utf-8') as file:
            file.write(f'epoch {ep}\n')
            file.write(evaluate_print)

        tools_get_logger('train').info(evaluate_print)
        evaluate_summary = [ep, train_loss, test_loss, novelty, div1, div2, gram2, gram3, gram4, bleu2, bleu3, bleu4]
        tools_write_log_to_file(config_train_generator.evaluate_log_format, evaluate_summary, f'{log_dir}/evaluate.log')


        if config_train_generator.is_save_model and ep >= 30:
            save_dir = f'{log_dir}/model_state/'
            tools_make_dir(save_dir)
            save_path = f'{save_dir}epoch_{ep}_{tools_get_time()}.pt'
            torch.save(seq2seq.state_dict(), save_path)
            tools_get_logger('train').info(
                f"epoch {ep} saving model {save_path}")

    tools_get_logger('train').info(f"{epoch_num} epochs done \n")


if __name__ == '__main__':
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
    elif args.dataset == 'acl' or args.dataset == 'expand':
        word2idx, idx2word, topic2idx, idx2topic, (train_essay, train_topic, train_mem), (
        test_essay, test_topic, test_mem) \
            = read_acl_origin_data(is_expand=args.dataset == 'expand')
        args.dataset = 'acl'
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


    writer, log_dir, start_time = tools_get_tensorboard_writer(dir_pre=writer_pre)

    train_generator_process(epoch_num=args.epoch,
                            batch_size=args.batch,
                            writer_logdir_starttime=(writer, log_dir, start_time),
                            train_all_dataset=train_all_dataset,
                            test_all_dataset=test_all_dataset,
                            seq2seq=seq2seq)
    writer.flush()
    writer.close()
