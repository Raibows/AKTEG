import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import random
from data import ZHIHU_dataset
from neural import KnowledgeEnhancedSeq2Seq, init_param
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_make_dir, tools_copy_file, tools_to_gpu
from preprocess import k_fold_split
from config import config_zhihu_dataset, config_train, config_seq2seq, config_concepnet


tools_setup_seed(667)
# device = torch.device(config_train.device_name)
device = torch.device('cuda:0')
tools_get_logger('train').info(f"using device {config_train.device_name}")

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

test_all_dataloader = DataLoader(test_all_dataset, batch_size=config_train.batch_size,
                                 num_workers=config_train.dataloader_num_workers, pin_memory=True)

tools_get_logger('train').info(f"load train data {len(train_all_dataset)} test data {len(test_all_dataset)} "
                               f"test/all {len(test_all_dataset)/(len(test_all_dataset)+len(train_all_dataset)):.4f}")


seq2seq = KnowledgeEnhancedSeq2Seq(vocab_size=len(train_all_dataset.word2idx),
                                   embed_size=config_seq2seq.embedding_size,
                                   pretrained_wv_path=config_seq2seq.pretrained_wv_path['tencent'],
                                   encoder_lstm_hidden=config_seq2seq.encoder_lstm_hidden_size,
                                   encoder_bid=config_seq2seq.encoder_lstm_is_bid,
                                   lstm_layer=config_seq2seq.lstm_layer_num,
                                   attention_size=config_seq2seq.attention_size,
                                   device=device)

init_param(seq2seq, init_way=config_train.model_init_way)
if config_train.is_load_model:
    seq2seq.load_state_dict(torch.load(config_seq2seq.model_load_path, map_location=device))
seq2seq.to(device)
seq2seq.eval()

optimizer = optim.AdamW(seq2seq.parameters(), lr=config_train.learning_rate)
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=train_all_dataset.word2idx['<pad>']).to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=3, min_lr=6e-6)
warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1e-2 if ep < 3 else 1)



def train(epoch, train_all_dataset, dataset_loader, teacher_force_ratio):
    seq2seq.train()
    loss_mean = 0.0
    with tqdm(total=len(dataset_loader), desc=f'train{epoch}') as pbar:
        for i, (topic, topic_len, mems, essay_input, essay_target, _) in enumerate(dataset_loader):

            topic, topic_len, mems, essay_input, essay_target = \
                tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, device=device)

            logits = seq2seq.forward(topic, topic_len, essay_input, mems, teacher_force_ratio=teacher_force_ratio)
            # [batch, essay_max_len, essay_vocab_size]

            # if (i+1) % 10 == 0:
            #     pass
                # sample = logits[0]
                # idx = sample.argmax(dim=-1)
                # idxs = torch.multinomial(torch.softmax(sample, dim=-1), num_samples=1)
                # idx = idx.squeeze()[:10].tolist()
                # idxs = idxs.squeeze()[:10].tolist()
                # tools_get_logger('argmax').info(f"\n{train_all_dataset.convert_idx2word(idx)}")
                # tools_get_logger('multin').info(f"\n{train_all_dataset.convert_idx2word(idxs)}")



            logits = logits.view(-1, len(train_all_dataset.word2idx))
            optimizer.zero_grad()
            loss = criterion(logits, essay_target.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=config_train.grad_clip_max_norm,
                                     norm_type=config_train.grad_clip_norm_type)
            optimizer.step()
            pbar.set_postfix_str(f"loss: {loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)

    return loss_mean / len(dataset_loader)

@torch.no_grad()
def validation(epoch, train_all_dataset, dataset_loader, prediction_path=None):
    seq2seq.eval()
    loss_mean = 0.0
    teacher_force_ratio = 0.0
    optimizer.zero_grad()
    show_interval = len(dataset_loader) // 5
    predicts_set = []
    topics_set = []
    original_essays_set = []


    with tqdm(total=len(dataset_loader), desc=f'validation{epoch}') as pbar:
        for i, (topic, topic_len, mems, essay_input, essay_target, _) in enumerate(dataset_loader):
            pass
            topic, topic_len, mems, essay_input, essay_target = \
                tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, device=device)

            logits = seq2seq.forward(topic, topic_len, essay_input, mems, teacher_force_ratio=teacher_force_ratio)
            # [batch, essay_len, vocab_size]
            if (i+1) % show_interval == 0:
                sample = logits[random.randint(0, logits.shape[0]-1)]
                idx = sample.argmax(dim=-1)
                idx = idx.squeeze()[:10].tolist()
                tools_get_logger('example').info(f"\n{train_all_dataset.convert_idx2topic(topic[0].tolist())}\n"
                                                 f"{train_all_dataset.convert_idx2word(idx)}\n")

            if prediction_path:
                predicts = logits.argmax(dim=-1)
                predicts_set.extend(predicts.tolist())
                topics_set.extend(topic.tolist())
                original_essays_set.extend(essay_target.tolist())


            loss = criterion(logits.view(-1, len(train_all_dataset.word2idx)), essay_target.view(-1)).item()
            loss_mean += loss
            pbar.set_postfix_str(f"loss: {loss:.4f}")
            pbar.update(1)

    if prediction_path:
        tools_make_dir(prediction_path)
        with open(prediction_path, 'w', encoding='utf-8') as file:
            for t, o, p in zip(topics_set, original_essays_set, predicts_set):
                t = train_all_dataset.convert_idx2word(t, sep=' ')
                o = train_all_dataset.convert_idx2word(o, sep='')
                p = train_all_dataset.convert_idx2word(p, sep='')

                file.write(f'{t}\n{o}\n{p}\n')
                file.write('--' * train_all_dataset.essay_padding_len)
                file.write('\n')

        tools_get_logger('validation').info(f"predictions epoch{epoch} done! the res is in {prediction_path}")

    return loss_mean / len(dataset_loader)




if __name__ == '__main__':
    writer, log_dir = tools_get_tensorboard_writer()
    best_save_loss = int(1e9)
    best_save_path = 'no_saved'
    begin_teacher_force_ratio = config_seq2seq.teacher_force_rate
    train_start_time = tools_get_time()
    for ep in range(config_train.epoch):
        kfolds = k_fold_split(train_all_dataset, config_train.batch_size, k=config_train.fold_k)
        train_loss = 0.0
        valid_loss = 0.0
        valid_loss_t = 0.0
        if ep >= 9:
            begin_teacher_force_ratio *= 0.96
            begin_teacher_force_ratio = max(begin_teacher_force_ratio, 0.75)

        for fold_no, (train_dataloader, valid_dataloader) in enumerate(kfolds):
            train_loss_t = train(ep, train_all_dataset, train_dataloader, begin_teacher_force_ratio)
            train_loss += train_loss_t
            if valid_dataloader:
                valid_loss_t = validation(train_all_dataset, valid_dataloader)
                valid_loss += valid_loss_t
            tools_get_logger('train').info(f'epoch {ep} fold {fold_no} done '
                                           f'train_loss {train_loss_t:.4f} valid_loss {valid_loss_t:.4f}')

        prediction_path = f'{log_dir}/epoch_{ep}.predictions'
        test_loss = validation(ep, train_all_dataset, test_all_dataloader,
                               prediction_path=prediction_path)
        # train_all_dataset.shuffle_memory()

        scheduler.step(test_loss)
        warmup_scheduler.step()

        train_loss /= len(kfolds)
        valid_loss /= len(kfolds)
        if valid_loss > 0.0:
            writer.add_scalar('Loss/valid', valid_loss, ep)
        writer.add_scalar('Loss/train', train_loss, ep)
        writer.add_scalar('Loss/test', test_loss, ep)
        del kfolds
        with open(prediction_path, 'a', encoding='utf-8') as file:
            file.write(f'epoch {ep}\ntrain_loss{train_loss}\nvalid_loss{valid_loss}\ntest_loss{test_loss}')

        tools_get_logger('train').info(f'epoch {ep} done train_loss {train_loss:.4f} '
                                       f'valid_loss {valid_loss:.4f} test_loss {test_loss:.4f}')

        if config_train.is_save_model and test_loss < best_save_loss:
            save_path = config_seq2seq.model_save_fmt.format(train_start_time, ep, test_loss)
            best_save_path = save_path
            if best_save_loss == int(1e9):
                tools_make_dir(save_path)
                tools_copy_file('./config.py', save_path+'.config.py')
            torch.save(seq2seq.state_dict(), save_path)
            best_save_loss = test_loss
            tools_get_logger('train').info(f"epoch {ep} saving model to {save_path}, now best_test_loss {best_save_loss:.4f}")

    tools_get_logger('train').info(f"{config_train.epoch} epochs done \n"
                                   f"the best model has saved to {best_save_path} \n"
                                   f"the prediction ")
    writer.flush()
    writer.close()