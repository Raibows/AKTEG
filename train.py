import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import ZHIHU_dataset
from neural import Encoder, Decoder, Seq2Seq, Memory_neural, uniform_init_weights
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_make_dir, tools_copy_file, tools_to_gpu
from preprocess import k_fold_split
from config import config_zhihu_dataset, config_train, config_seq2seq, config_concepnet


tools_setup_seed(667)

device = torch.device(config_train.device_name)
tools_get_logger('train').info(f"using device {config_train.device_name}")

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
                                 prior=train_all_dataset.get_prior())

test_all_dataloader = DataLoader(test_all_dataset, batch_size=config_train.batch_size,
                                 num_workers=config_train.dataloader_num_workers, pin_memory=True)

tools_get_logger('train').info(f"load train data {len(train_all_dataset)} test data {len(test_all_dataset)}")

encoder = Encoder(embed_size=config_seq2seq.embedding_size,
                  layer_num=config_seq2seq.encoder_lstm_layer_num,
                  hidden_size=config_seq2seq.encoder_lstm_hidden_size,
                  is_bid=config_seq2seq.encoder_lstm_is_bid)

decoder = Decoder(vocab_size=train_all_dataset.word_vocab_size,
                  embed_size=config_seq2seq.embedding_size,
                  layer_num=config_seq2seq.decoder_lstm_layer_num,
                  encoder_output_size=encoder.output_size,)

memory_neural = Memory_neural(embed_size=config_seq2seq.embedding_size,
                              decoder_hidden_size=decoder.hidden_size)

seq2seq = Seq2Seq(encoder=encoder,
                  decoder=decoder,
                  memory_neural=memory_neural,
                  topic_padding_num=config_zhihu_dataset.topic_padding_num,
                  total_vocab_size=train_all_dataset.word_vocab_size,
                  embed_size=config_seq2seq.embedding_size,
                  pretrained_path=config_seq2seq.pretrained_wv_path,
                  attention_size=config_seq2seq.attention_size,
                  device=device)
seq2seq.apply(uniform_init_weights)
if config_train.is_load_model:
    seq2seq.load_state_dict(torch.load(config_seq2seq.model_load_path, map_location=device))
seq2seq.to(device)
seq2seq.eval()

optimizer = optim.AdamW(seq2seq.parameters(), lr=config_train.learning_rate, weight_decay=0.3)
criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=3, min_lr=6e-6)
warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 1e-2 if ep < 3 else 1)




def train(train_all_dataset, dataset_loader, teacher_force_ratio):
    seq2seq.train()
    loss_mean = 0.0
    with tqdm(total=len(dataset_loader), desc='train') as pbar:
        for topic, topic_len, mems, essay_input, essay_target, _ in dataset_loader:
            topic, topic_len, mems, essay_input, essay_target = \
                tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, device=device)

            logits = seq2seq.forward((topic, topic_len+1), essay_input, mems, teacher_force_ratio=teacher_force_ratio)
            logits = logits.permute(1, 0, 2) # [batch, essay_max_len, essay_vocab_size]

            essay_target = essay_target.view(-1)
            logits = logits.reshape(-1, train_all_dataset.word_vocab_size)
            optimizer.zero_grad()
            loss = criterion(logits, essay_target)
            loss.backward()
            idx = logits[:10].argmax(dim=1).tolist()
            tools_get_logger('example').info(f"{train_all_dataset.convert_idx2words(idx)}")
            nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=config_train.grad_clip_max_norm,
                                     norm_type=config_train.grad_clip_norm_type)
            optimizer.step()

            pbar.set_postfix_str(f"loss: {loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)
    return loss_mean / len(dataset_loader)

@torch.no_grad()
def validation(train_all_dataset, dataset_loader, prediction_path=None):
    seq2seq.eval()
    loss_mean = 0.0
    teacher_force_ratio = 0.0
    optimizer.zero_grad()
    predicts_set = []
    topics_set = []
    original_essays_set = []
    with tqdm(total=len(dataset_loader), desc='validation') as pbar:
        for topic, topic_len, mems, essay_input, essay_target, _ in dataset_loader:

            topic, topic_len, mems, essay_input, essay_target = \
                tools_to_gpu(topic, topic_len, mems, essay_input, essay_target, device=device)

            logits = seq2seq.forward((topic, topic_len+1), essay_input, mems=mems, teacher_force_ratio=teacher_force_ratio)
            logits = logits.permute(1, 0, 2) # [batch, essay_max_len, essay_vocab_size]
            if prediction_path:
                predicts = logits.argmax(dim=2)
                predicts_set.extend(predicts.tolist())
                topics_set.extend(topic.tolist())
                original_essays_set.extend(essay_target.tolist())


            essay_target = essay_target.view(-1)
            logits = logits.reshape(-1, train_all_dataset.word_vocab_size)
            loss = criterion(logits, essay_target)

            pbar.set_postfix_str(f"loss: {loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)

    if prediction_path:
        tools_make_dir(prediction_path)
        with open(prediction_path, 'w', encoding='utf-8') as file:
            for t, o, p in zip(topics_set, original_essays_set, predicts_set):
                t, o, p = train_all_dataset.convert_idx2words(t), train_all_dataset.convert_idx2words(o), \
                          train_all_dataset.convert_idx2words(p)
                file.write(' '.join(t))
                file.write('\n')
                file.write(' '.join(o))
                file.write('\n')
                file.write(' '.join(p))
                file.write('\n')

        tools_get_logger('validation').info(f"predictions done! the res is in {prediction_path}")

    return loss_mean / len(dataset_loader)




if __name__ == '__main__':
    writer, log_dir = tools_get_tensorboard_writer()
    best_save_loss = 1e9
    # test_loss = validation(train_all_dataset, test_all_dataloader)
    begin_teacher_force_ratio = config_seq2seq.teacher_force_rate
    for ep in range(config_train.epoch):
        kfolds = k_fold_split(train_all_dataset, config_train.batch_size, k=config_train.fold_k)
        train_loss = 0.0
        valid_loss = 0.0
        valid_loss_t = 0.0
        if ep >= 9: begin_teacher_force_ratio *= 0.95
        for fold_no, (train_dataloader, valid_dataloader) in enumerate(kfolds):
            train_loss_t = train(train_all_dataset, train_dataloader, begin_teacher_force_ratio)
            if valid_dataloader:
                valid_loss_t = validation(train_all_dataset, valid_dataloader)
                valid_loss += valid_loss_t
            train_loss += train_loss_t
            tools_get_logger('train').info(f'epoch {ep} fold {fold_no} done '
                                           f'train_loss {train_loss_t:.4f} valid_loss {valid_loss_t:.4f}')

        test_loss = validation(train_all_dataset, test_all_dataloader,
                               prediction_path=f'{log_dir}/epoch_{ep}.predictions')
        train_all_dataset.shuffle_memory()

        scheduler.step(test_loss)
        warmup_scheduler.step()

        train_loss /= len(kfolds)
        valid_loss /= len(kfolds)
        del kfolds
        writer.add_scalar('Loss/train', train_loss, ep)
        writer.add_scalar('Loss/valid', valid_loss, ep)
        writer.add_scalar('Loss/test', test_loss, ep)

        tools_get_logger('train').info(f'epoch {ep} done train_loss {train_loss:.4f} '
                                       f'valid_loss {valid_loss:.4f} test_loss {test_loss:.4f}')

        if config_train.is_save_model and test_loss < best_save_loss:
            save_path = config_seq2seq.model_save_fmt.format(tools_get_time(), test_loss)
            tools_make_dir(save_path)
            tools_copy_file('./config.py', save_path+'.config.py')
            torch.save(seq2seq.state_dict(), save_path)
            best_save_loss = test_loss
            tools_get_logger('train').info(f"saving model to {save_path}, now best_test_loss {best_save_loss:.4f}")








    writer.flush()
    writer.close()