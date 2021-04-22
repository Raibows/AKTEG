import torch
import torch.nn as nn
from tools import tools_load_pickle_obj

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, hidden_size, is_bid, pretrained_path):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        if pretrained_path:
            self.embedding_layer.from_pretrained(
                torch.tensor(tools_load_pickle_obj(pretrained_path), dtype=torch.float)
            )
        self.embedding_layer.weight.requires_grad = True
        self.lstm = nn.LSTM(embed_size, hidden_size, layer_num, bidirectional=is_bid)
        self.direction = 2 if is_bid else 1
        self.output_size = self.direction * self.hidden_size

    def forward(self, *inputs):
        # (sen, sen_len)
        embeddings = self.embedding_layer(inputs[0].permute(1, 0))
        embeddings = torch.relu(embeddings)

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
    def __init__(self, vocab_size, embed_size, layer_num, hidden_size, pretrained_path):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        if pretrained_path:
            self.embedding_layer.from_pretrained(
                torch.tensor(tools_load_pickle_obj(pretrained_path), dtype=torch.float)
            )
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
        embeddings = torch.relu(embeddings)

        outs, (h, c) = self.lstm(embeddings, (init_h, init_c))

        logits = self.fc(outs.squeeze(0))

        return logits, (h, c)

class Memory_neural(nn.Module):
    def __init__(self, vocab_size, embed_size, decoder_hidden_size, pretrained_path):
        super(Memory_neural, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        if pretrained_path:
            self.embedding_layer.from_pretrained(
                torch.tensor(tools_load_pickle_obj(pretrained_path), dtype=torch.float)
            )
        self.embedding_layer.weight.requires_grad = True
        self.W = nn.Linear(decoder_hidden_size, embed_size, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, decoder_hidden_s_t_1, mems):
        mems = mems.permute(1, 0)
        embeddings = self.embedding_layer(mems).permute(1, 0, 2)
        embeddings = self.dropout(embeddings)
        # embeddings [batch, len, embed_size]
        v_t = torch.tanh(self.W(decoder_hidden_s_t_1))
        # torch.bmm()
        # v_t here is a column vector
        q_t = torch.softmax(embeddings @ v_t.unsqueeze(2), dim=1)
        # q_t here is a column vector
        m_t = q_t.permute(0, 2, 1) @ embeddings

        return m_t


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, essay_vocab_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.essay_vocab_size = essay_vocab_size
        self.device = device

    def forward(self, topic_len_input:tuple, essay_input:torch.Tensor, teacher_force_ratio=0.5):

        # topic_input [topic, topic_len]
        # topic [batch_size, seq_len]
        teacher_force_ratio = torch.tensor(teacher_force_ratio, dtype=torch.float, device=self.device)
        batch_size = topic_len_input[0].shape[0]
        max_essay_len = essay_input.shape[1]

        decoder_outputs = torch.zeros([max_essay_len, batch_size, self.essay_vocab_size], device=self.device)

        h, c = self.encoder(topic_len_input[0], topic_len_input[1])

        teacher_mode_chocie = torch.rand([max_essay_len], device=self.device)
        # first input token is <sos>
        now_input = essay_input[:, 0]
        for now_step in range(1, max_essay_len):
            logits, (h, c) = self.decoder(now_input, h, c)
            decoder_outputs[now_step - 1] = logits
            if teacher_mode_chocie[now_step] < teacher_force_ratio:
                now_input = essay_input[:, now_step]
            else:
                now_input = logits.argmax(1)

        logits, _ = self.decoder(now_input, h, c)
        decoder_outputs[-1] = logits

        return decoder_outputs

if __name__ == '__main__':
    word_num = 100
    maxlen = 5
    batch_size = 64
    word_dim = 300
    num_hidden = 200

    # sentence = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    # sentence_len = torch.randint(1, maxlen, size=[batch_size], dtype=torch.long)
    #
    #
    # encoder = Encoder(word_num, embed_size=word_dim, layer_num=2, hidden_size=num_hidden, is_bid=True)
    # # h, c = encoder(sentence, sentence_len)
    #
    # decoder = Decoder(word_num, word_dim, 2, encoder.output_size)
    #
    # # logits = decoder((sentence, sentence_len), h, c)
    #
    # seq2seq = Seq2Seq(encoder, decoder, word_num, torch.device('cpu'))
    #
    # seq2seq.forward((sentence, sentence_len), sentence)
    decoder_hidden_size = 100
    mem_per_sample = 20
    mems = torch.randint(0, word_num, [batch_size, mem_per_sample])
    memory_neural = Memory_neural(word_num, word_dim, decoder_hidden_size, None)
    decoder_hidden = torch.rand([batch_size, decoder_hidden_size])

    m_t = memory_neural.forward(decoder_hidden, mems)

    print(m_t.shape)



    pass