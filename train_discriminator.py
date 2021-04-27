import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import random
from data import ZHIHU_dataset, InputLabel_dataset
from neural import KnowledgeEnhancedSeq2Seq, CNNDiscriminator, init_param
from tools import tools_get_logger, tools_get_tensorboard_writer, tools_get_time, \
    tools_setup_seed, tools_make_dir, tools_copy_file, tools_to_gpu
from preprocess import k_fold_split
from config import config_zhihu_dataset, config_train_generator, config_seq2seq, \
    config_concepnet, config_train_public, config_wordcnn, config_train_discriminator
from metric import TrainDiscriminatorMetric


tools_setup_seed(667)
device = torch.device(config_train_public.device_name)
# device = torch.device('cuda:0')
tools_get_logger('train_D').info(f"using device {config_train_public.device_name}")


def train_discriminator(ep, discriminator, train_dataloader, criterion, optimizer, metric):
    discriminator.train()
    with tqdm(total=len(train_dataloader), desc=f'train_D {ep}') as pbar:
        for input, label in train_dataloader:
            input, label = tools_to_gpu(input, label, device=device)
            logits = discriminator.forward(input)
            loss = criterion.forward(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f'loss: {loss.item():.4f}')
            metric(logits, label, loss.item())
            pbar.update(1)

    acc_all, acc_fake, acc_real, loss_mean = metric.value()
    tools_get_logger('test_D').info(
        f'epoch {ep} done! acc_all {acc_all:.4f}, acc_fake {acc_fake:.4f}, acc_real {acc_real:.4f}, loss_mean {loss_mean:.4f}'
    )
    return metric.value()

@torch.no_grad()
def test_discriminator(ep, discriminator, test_dataloader, criterion, metric):
    discriminator.eval()
    with tqdm(total=len(test_dataloader), desc=f'test_D {ep}') as pbar:
        for input, label in test_dataloader:
            input, label = tools_to_gpu(input, label, device=device)
            logits = discriminator.forward(input)
            loss = criterion.forward(logits, label)
            metric(logits, label, loss.item())
            acc_all, acc_fake, acc_real, loss_mean = metric.value()
            pbar.set_postfix_str(f'acc_all{acc_all:.3f} acc_fake{acc_fake:.3f} acc_real{acc_real:.3f} loss{loss_mean:.3f}')
            pbar.update(1)

    acc_all, acc_fake, acc_real, loss_mean = metric.value()
    tools_get_logger('test_D').info(
        f'epoch {ep} done! acc_all {acc_all:.4f}, acc_fake {acc_fake:.4f}, acc_real {acc_real:.4f}, loss_mean {loss_mean:.4f}'
    )
    return metric.value()

def train_discriminator_process(writer, train_all_dataset, epoch, batch_size, generator, discriminator):
    from config import config_train_discriminator as ctd
    start_time = tools_get_time()
    generator.eval()
    train_all_dataloader = DataLoader(train_all_dataset, batch_size=ctd.generate_batch_size)
    num_all = min(ctd.generate_batch_num * ctd.generate_batch_size, len(train_all_dataset))
    num_all *= 2 # half is fake generated, half is real origin
    label_dict = ['fake'] + list(train_all_dataset.topic2idx.keys())
    label_dict = {l:i for i, l in enumerate(label_dict)}
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(discriminator.parameters(), lr=ctd.learning_rate)
    metric = TrainDiscriminatorMetric(label_dict['fake'])

    @torch.no_grad()
    def prepare_datas(label_smooth=0.9, eps=0.1):
        labels_set = []
        datas_set = []
        def process_label(topic_idxs_batch, fake=False):
            batch = []
            fake = 1.0 if fake else eps
            for b in topic_idxs_batch:
                temp = [eps for _ in label_dict]
                temp[label_dict['fake']] = fake
                topic_words = train_all_dataset.convert_idx2topic(b)
                for topic in topic_words:
                    if topic in label_dict: temp[label_dict[topic]] = 1.0
                temp = [x * label_smooth for x in temp]
                batch.append(temp)
            return batch
        with tqdm(total=(num_all/ctd.generate_batch_size)//2+1, desc=f'generate') as pbar:
            for i, (topic, topic_len, mems, essay_input, essay_target, _) in enumerate(train_all_dataloader):
                topic, topic_len, mems, essay_input = tools_to_gpu(topic, topic_len, mems, essay_input, device=device)
                logits = generator.forward(topic, topic_len, essay_input, mems, teacher_force_ratio=0.0)
                # [batch, essay_len, vocab_size]
                predicts = logits.argmax(dim=-1)
                datas_set.extend(predicts.tolist())
                labels_set.extend(process_label(topic.tolist(), fake=True))
                datas_set.extend(essay_target.tolist())
                labels_set.extend(process_label(topic.tolist(), fake=False))
                pbar.update(1)
                if len(labels_set) == num_all: break

        return datas_set, labels_set

    datas_set, labels_set = prepare_datas(label_smooth=ctd.label_smooth, eps=ctd.label_eps)
    dataset = InputLabel_dataset(datas_set, labels_set)
    test_num = int(len(dataset) * ctd.test_data_split_ratio)
    train_set, test_set = random_split(dataset, [len(dataset)-test_num, test_num])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                             num_workers=config_train_public.dataloader_num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size,
                                 num_workers=config_train_public.dataloader_num_workers)

    best_acc = 0.0
    for ep in range(epoch):
        acc_all, acc_fake, acc_real, loss_mean = train_discriminator(ep, discriminator, train_dataloader, criterion, optimizer, metric)
        writer.add_scalar('Train/acc_all', acc_all, ep)
        writer.add_scalar('Train/acc_fake', acc_fake, ep)
        writer.add_scalar('Train/acc_real', acc_real, ep)
        writer.add_scalar('Train/loss', loss_mean, ep)
        metric.reset()
        acc_all, acc_fake, acc_real, loss_mean = test_discriminator(ep, discriminator, test_dataloader, criterion, metric)
        writer.add_scalar('Test/acc_all', acc_all, ep)
        writer.add_scalar('Test/acc_fake', acc_fake, ep)
        writer.add_scalar('Test/acc_real', acc_real, ep)
        writer.add_scalar('Test/loss', loss_mean, ep)
        metric.reset()
        if acc_all > best_acc and ctd.is_save_model:
            best_acc = acc_all
            save_path = config_wordcnn.model_save_fmt.format(start_time, ep, acc_all)
            tools_make_dir(save_path)
            torch.save(discriminator.state_dict(), save_path)
            tools_get_logger('train_D').info(
                f'epoch {ep} now best acc_all {best_acc:.5f}, saving model to {save_path}'
            )

if __name__ == '__main__':
    assert config_seq2seq.model_load_path != None

    train_all_dataset = ZHIHU_dataset(path=config_zhihu_dataset.train_data_path,
                                      topic_num_limit=config_zhihu_dataset.topic_num_limit,
                                      topic_padding_num=config_zhihu_dataset.topic_padding_num,
                                      vocab_size=config_zhihu_dataset.vocab_size,
                                      essay_padding_len=config_zhihu_dataset.essay_padding_len,
                                      prior=None, encode_to_tensor=True,
                                      mem_corpus_path=config_concepnet.memory_corpus_path)
    train_all_dataset.print_info()
    seq2seq = KnowledgeEnhancedSeq2Seq(vocab_size=len(train_all_dataset.word2idx),
                                       embed_size=config_seq2seq.embedding_size,
                                       pretrained_wv_path=config_seq2seq.pretrained_wv_path['tencent'],
                                       encoder_lstm_hidden=config_seq2seq.encoder_lstm_hidden_size,
                                       encoder_bid=config_seq2seq.encoder_lstm_is_bid,
                                       lstm_layer=config_seq2seq.lstm_layer_num,
                                       attention_size=config_seq2seq.attention_size,
                                       device=device)
    seq2seq.load_state_dict(torch.load(config_seq2seq.model_load_path, map_location=device))
    seq2seq.to(device)
    seq2seq.eval()

    discriminator = CNNDiscriminator(label_num=len(train_all_dataset.topic2idx)+1,
                                     vocab_size=len(train_all_dataset.word2idx),
                                     embed_size=config_wordcnn.embed_size,
                                     channel_nums=config_wordcnn.channel_nums,
                                     kernel_sizes=config_wordcnn.kernel_sizes)
    if config_wordcnn.model_load_path:
        discriminator.load_state_dict(torch.load(config_seq2seq.model_load_path, map_location=device))
    discriminator.to(device)
    discriminator.eval()

    writer, log_dir = tools_get_tensorboard_writer(dir_pre='train_D')

    train_discriminator_process(train_all_dataset=train_all_dataset,
                                epoch=config_train_discriminator.epoch,
                                batch_size=config_train_discriminator.batch_size,
                                generator=seq2seq, discriminator=discriminator,
                                writer=writer)
