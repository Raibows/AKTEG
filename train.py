import torch
import numpy as np
import random
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from data import ZHIHU_dataset
from neural import Encoder, Decoder, Seq2Seq
from tools import tools_get_logger, tools_get_tensorboard_writer

def setup_seed(seed):
    print(f'setup seed {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(667)
device = torch.device('cuda:0')

dataset = ZHIHU_dataset(path='../data/zhihu.txt', topic_num_limit=100, essay_vocab_size=50000,
                        topic_threshold=4, topic_padding_num=5, essay_padding_len=100)
dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)

encoder = Encoder(dataset.topic_num_limit, 100, layer_num=2, hidden_size=100, is_bid=True)
decoder = Decoder(dataset.essay_vocab_size, 100, 2, encoder.output_size)
seq2seq = Seq2Seq(encoder, decoder, dataset.essay_vocab_size, device)
seq2seq.to(device)

optimizer = optim.AdamW(seq2seq.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

def to_gpu(*params, device=torch.device('cpu')):
    return [p.to(device) for p in params]


def train():
    seq2seq.train()
    loss_mean = 0.0
    with tqdm(total=len(dataset_loader), desc='train') as pbar:
        for topic, topic_len, essay, essay_len in dataset_loader:
            temp_sos = torch.full([essay.size(0), 1], dataset.essay2idx['<sos>'])
            temp_eos = torch.full([essay.size(0), 1], dataset.essay2idx['<eos>'])
            essay_input = torch.cat([temp_sos, essay], dim=1)
            essay_target = torch.cat([essay, temp_eos], dim=1)
            essay_len += 1
            topic, topic_len, essay_input, essay_target, essay_len = \
                to_gpu(topic, topic_len, essay_input, essay_target, essay_len, device=device)

            logits = seq2seq.forward((topic, topic_len), (essay_input, essay_len), (essay_target, essay_len), teacher_force_ratio=1.0)

            essay_target = essay_target.view(-1)
            logits = logits.view(-1, dataset.essay_vocab_size)
            optimizer.zero_grad()
            loss = criterion(logits, essay_target)
            loss.backward()
            nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=1.0, norm_type=2.0)
            optimizer.step()

            pbar.set_postfix_str(f"{loss.item():.4f}")
            loss_mean += loss.item()
            pbar.update(1)
    return loss_mean / len(dataset_loader)



if __name__ == '__main__':
    writer = tools_get_tensorboard_writer()
    for ep in range(100):
        train_loss = train()
        writer.add_scalar('Loss/train', train_loss, ep)







