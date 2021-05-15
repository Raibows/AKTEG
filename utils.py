import random

import torch
import torch.nn as nn
from tools import tools_load_pickle_obj


class CNNDiscriminator(nn.Module):
    def __init__(self, label_num, vocab_size, embed_size, channel_nums: list,
                 kernel_sizes: list):
        super(CNNDiscriminator, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.embedding_layer.requires_grad_()

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.convs = nn.ModuleList()
        for c, k in zip(channel_nums, kernel_sizes):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=embed_size, out_channels=c,
                              kernel_size=k),
                    nn.BatchNorm1d(c)
                )
            )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(channel_nums), label_num)

    def forward(self, inputs_x):
        embeddings = self.dropout(self.embedding_layer(inputs_x))
        embeddings = embeddings.permute(0, 2, 1)
        # [batch, embed_size, seq_len]
        outs = torch.cat(
            [self.pool(torch.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1
        )
        outs = self.dropout(outs)

        logits = self.fc(outs)

        return logits

class Encoder_LSTM(nn.Module):
    def __init__(self, embed_size, layer_num, hidden_size, is_bid, is_cat=True, batch_first=False):
        super(Encoder_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lstm = nn.LSTM(embed_size, hidden_size, layer_num, bidirectional=is_bid,
                            dropout=0.5 if layer_num > 1 else 0.0, batch_first=batch_first)
        self.direction = 2 if is_bid else 1
        if is_cat:
            self.output_size = self.direction * self.hidden_size
        else:
            self.output_size = self.hidden_size
        self.is_cat = is_cat

    def forward(self, *inputs):
        # (sen_embeddings, sen_len)
        # [batch, len, embed]

        sort = torch.sort(inputs[1], descending=True)
        sent_len_sort, idx_sort = sort.values, sort.indices
        idx_reverse = torch.argsort(idx_sort)

        sent = inputs[0].index_select(0, idx_sort)

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sort.cpu(), batch_first=True)
        # outs, (h, c) = self.lstm(inputs[0])
        outs, (h, c) = self.lstm.forward(sent_packed)

        outs, _ = nn.utils.rnn.pad_packed_sequence(outs, padding_value=0.0, batch_first=True)
        #
        outs = outs.index_select(0, idx_reverse)
        outs_fw, outs_bw = outs.split(outs.size(-1) // 2, dim=-1)
        h, c = h.index_select(1, idx_reverse), c.index_select(1, idx_reverse)
        h, c = h.view(1, self.direction, h.size(1), -1), c.view(1, self.direction, c.size(1), -1)
        # [layer, direction, batch, -1]
        if self.is_cat:
            h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)
            c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)
            outs = torch.cat([outs_fw, outs_bw], dim=-1)
        else:
            h, c = h[:, 0, :, :] + h[:, 1, :, :], c[:, 0, :, :] + c[:, 1, :, :]
            outs = outs_fw + outs_bw


        return outs, (h, c)

class Attention(nn.Module):
    def __init__(self, enc_output_size, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_output_size + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_ouputs, dec_hidden, enc_mask=None):
        # encoder_outputs [batch, topic_num, enc_output_size]
        # dec_hidden [1, batch, dec_dim]
        # enc_mask = [batch, topic_num, 1]
        batch, topic_num, _ = encoder_ouputs.shape
        dec_hidden = dec_hidden.squeeze().unsqueeze(1).repeat(1, topic_num, 1)  # [batch, topic_num, dec_dim]
        energy = torch.cat([encoder_ouputs, dec_hidden], dim=2)
        energy = torch.tanh(self.attn.forward(energy))
        attention = self.v.forward(energy)  # [batch, topic_num, 1]
        if enc_mask != None:
            enc_mask = enc_mask.squeeze()[:, :topic_num]
            attention = attention.masked_fill(enc_mask.unsqueeze(2) == False, -1e10)

        return torch.softmax(attention, dim=1)

class Decoder_proj(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, encoder_output_size):
        super(Decoder_proj, self).__init__()
        self.input_size = embed_size + encoder_output_size + embed_size  # decoder_embed + encoder_out + memory_embed
        self.layer_num = layer_num
        self.embed_size = embed_size
        self.hidden_size = encoder_output_size
        self.lstm = nn.LSTM(self.input_size, encoder_output_size, layer_num, batch_first=True,
                            bidirectional=False, dropout=0.5 if layer_num > 1 else 0.0)
        self.fc = nn.Linear(encoder_output_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_to_lstm, init_h, init_c):
        # [batch, 1, inputs_size]
        outs, (h, c) = self.lstm(input_to_lstm.unsqueeze(1), (init_h, init_c))

        logits = self.fc(outs.squeeze(0))

        return self.dropout(logits), (h, c)


def init_param(self, init_way=None):
    if hasattr(self, 'init_params'):
        self.init_params()
    else:
        if init_way == 'uniform':
            for param in self.parameters():
                if param.requires_grad:
                    nn.init.uniform_(param.data, -0.08, 0.08)
        elif init_way == 'xavier':
            for param in self.parameters():
                if param.requires_grad:
                    nn.init.xavier_normal(param.data)
        elif init_way == 'noraml':
            for param in self.parameters():
                if param.requires_grad:
                    nn.init.normal_(param.data, mean=0.0, std=0.08)
        elif init_way == 'kaiming':
            for param in self.parameters():
                if param.requires_grad:
                    nn.init.kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
    if hasattr(self, 'embedding_layer') and hasattr(self, 'pretrained_wv_path'):
        if self.pretrained_wv_path:
            # severe bug finally found at 5/13 15:10
            embeddings = torch.tensor(tools_load_pickle_obj(self.pretrained_wv_path))
            self.embedding_layer = nn.Embedding.from_pretrained(embeddings)
        self.embedding_layer.weight.requires_grad = True


def truncated_normal_(tensor, mean=0, std=0.05):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
