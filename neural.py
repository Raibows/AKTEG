import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, hidden_size, is_bid):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.embedding_layer.weight.requires_grad = True
        self.lstm = nn.LSTM(embed_size, hidden_size, layer_num, bidirectional=is_bid)
        self.direction = 2 if is_bid else 1
        self.output_size = self.direction * self.hidden_size

    def forward(self, *inputs):
        # (sen, sen_len)
        embeddings = self.embedding_layer(inputs[0].permute(1, 0))
        embeddings = F.relu(embeddings)

        sort = torch.sort(inputs[1], descending=True)
        sent_len_sort, idx_sort = sort.values, sort.indices
        idx_reverse = torch.argsort(idx_sort)

        sent = embeddings.index_select(1, idx_sort)

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sort.cpu())
        _, (h, c) = self.lstm(sent_packed)
        # outs, _ = nn.utils.rnn.pad_packed_sequence(outs, padding_value=-1e9)

        # outs = outs.index_select(1, idx_reverse)
        h = h.index_select(1, idx_reverse)
        h = h.reshape(self.layer_num, h.size(1), -1)
        c = c.index_select(1, idx_reverse)
        c = c.reshape(self.layer_num, c.size(1), -1)

        # select the real final state
        # outs = outs[inputs[1]-1, torch.arange(outs.size(1)), :]

        return h, c

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, hidden_size):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.embedding_layer.weight.requires_grad = True
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lstm = nn.LSTM(embed_size, hidden_size, layer_num, bidirectional=False)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, single_word, init_h, init_c):
        # single_word [batch, 1]
        if single_word.dim() == 1:
            single_word = single_word.unsqueeze(1)
        embeddings = self.embedding_layer(single_word.permute(1, 0))
        embeddings = F.relu(embeddings)

        outs, (h, c) = self.lstm(embeddings, (init_h, init_c))

        logits = self.fc(outs.squeeze(0))

        return logits, (h, c)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, essay_vocab_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.essay_vocab_size = essay_vocab_size
        self.device = device

    def forward(self, topic_len_input:tuple, essay_input:torch.Tensor, essay_target:torch.Tensor,
                teacher_force_ratio=0.5):

        # topic_input [topic, topic_len]
        # topic [batch_size, seq_len]
        batch_size = topic_len_input[0].shape[0]
        max_essay_len = essay_input.shape[1]

        decoder_outputs = torch.zeros([max_essay_len, batch_size, self.essay_vocab_size], device=self.device)

        h, c = self.encoder(topic_len_input[0], topic_len_input[1])

        for now_input in range(max_essay_len):
            logits, (h, c) = self.decoder(essay_input[:, now_input], h, c)
            decoder_outputs[now_input] = logits

        return decoder_outputs

if __name__ == '__main__':
    word_num = 100
    maxlen = 5
    batch_size = 4
    word_dim = 300
    num_hidden = 200

    sentence = torch.zeros([batch_size, maxlen], dtype=torch.long)
    sentence_len = torch.randint(1, maxlen, size=[batch_size], dtype=torch.long)


    encoder = Encoder(word_num, embed_size=word_dim, layer_num=2, hidden_size=num_hidden, is_bid=True)
    # h, c = encoder(sentence, sentence_len)

    decoder = Decoder(word_num, word_dim, 2, encoder.output_size)

    # logits = decoder((sentence, sentence_len), h, c)

    seq2seq = Seq2Seq(encoder, decoder, word_num, torch.device('cpu'))

    seq2seq.forward((sentence, sentence_len), sentence, sentence)


    pass