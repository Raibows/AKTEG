import torch
from torch import nn
from neural import Memory_neural

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, dropout, device):
        super(MultiheadAttention, self).__init__()
        assert output_dim % n_heads == 0
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor([output_dim / n_heads], dtype=torch.float, device=device))

    def forward(self, query, key, value, mask=None):
        # query [batch, len, temp]
        batch = query.shape[0]
        seqlen = query.shape[1]
        Q = self.W_q(query).reshape(batch, self.n_heads, seqlen, -1)
        K = self.W_k(key).reshape(batch, self.n_heads, seqlen, -1)
        V = self.W_v(value).reshape(batch, self.n_heads, seqlen, -1)

        attention = Q @ K.permute(0, 1, 3, 2) / self.scale #[batch, nhead, seq_len, seq_len]
        if mask != None:
            temp = torch.exp(attention).masked_fill(mask == False, 0.0)
            attention = temp / torch.sum(temp, dim=-1).unsqueeze(3)
            attention[attention.isnan()] = 0.0
            # attention.masked_fill(mask == False, -1e10)
        attention = self.dropout(attention)
        # attention = self.dropout(torch.softmax(attention, dim=-1))

        res = attention @ V #[batch, nhead, seqlen, temp]

        outs = res.permute(0, 2, 1, 3).contiguous().view(batch, seqlen, self.output_dim) # concat nheads

        outs = self.fc(outs)

        return outs, res.permute(1, 0, 2, 3)

class AttentionBasedEncoder(nn.Module):
    def __init__(self, embed_size, input_dim, output_dim, nheads, device):
        super(AttentionBasedEncoder, self).__init__()
        self.attention_layer = MultiheadAttention(input_dim, output_dim, nheads, 0.5, device)
        self.W_q = nn.Linear(embed_size, input_dim)
        self.W_k = nn.Linear(embed_size, input_dim)
        self.W_v = nn.Linear(embed_size, input_dim)
        self.W_h = nn.Linear(output_dim, output_dim)
        self.W_c = nn.Linear(output_dim, output_dim, bias=False)
        self.output_size = output_dim
        self.hidden_cell_size = output_dim // nheads


    def forward(self, topic_embeddings, mask):
        # [batch, seqlen, embed_size]
        query = self.W_q(topic_embeddings)
        key = self.W_k(topic_embeddings)
        value = self.W_v(topic_embeddings)
        outs, hiddens = self.attention_layer.forward(query, key, value, mask)
        # hiddens [nhead, batch, seqlen, temp]
        state = torch.sum(hiddens, dim=2) #[nhead, batch, output_dim]
        state = torch.tanh(state).unsqueeze(0)
        h, c = self.W_h(state), self.W_c(state)

        return outs, (h, c)

class Decoder2(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, encoder_output_size, encoder_hidden_cell_size):
        super(Decoder2, self).__init__()
        self.input_size = embed_size + encoder_output_size + embed_size # decoder_embed + encoder_out + memory_embed
        temp = self.input_size // 2
        self.sharp_linear = nn.Linear(self.input_size, temp)
        self.layer_num = layer_num
        self.embed_size = embed_size
        self.hidden_size = encoder_output_size
        self.input_size = temp
        self.lstm = nn.LSTM(self.input_size, encoder_hidden_cell_size, layer_num,
                            bidirectional=False, dropout=0.5 if layer_num > 1 else 0.0)
        self.fc = nn.Linear(encoder_output_size, vocab_size)
        self.dropout = nn.Dropout(0.5)


    def forward(self, input_to_lstm, init_h, init_c):
        input_to_lstm = input_to_lstm.unsqueeze(0) # [1-single-token, batch, input_size]
        input_to_lstm = self.sharp_linear.forward(input_to_lstm)
        outs, (h, c) = self.lstm(input_to_lstm, (init_h, init_c))

        logits = self.fc(outs.squeeze(0))

        return self.dropout(logits), (h, c)

class KnowledgeEnhancedAttentionSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, encoder_input_dim, encoder_output_dim, encoder_nheads,
                 attention_size, device, mask_idx):
        super(KnowledgeEnhancedAttentionSeq2Seq, self).__init__()
        self.pretrained_wv_path = pretrained_wv_path
        self.encoder = AttentionBasedEncoder(embed_size, encoder_input_dim, encoder_output_dim, encoder_nheads, device)
        self.decoder = Decoder2(vocab_size, embed_size, encoder_nheads,
                                self.encoder.output_size, self.encoder.hidden_cell_size)
        self.memory_neural = Memory_neural(embed_size, self.decoder.hidden_size)
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.W_1 = nn.Linear(self.encoder.output_size, attention_size, bias=False)
        self.W_2 = nn.Linear(self.decoder.hidden_size, attention_size, bias=False)
        self.mask_idx = torch.tensor(mask_idx, dtype=torch.int64, device=device)
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = nn.Dropout(0.5)

    def get_mask_matrix(self, idx):
        batch, maxlen = idx.shape
        mask = (idx != self.mask_idx).unsqueeze(1).expand(batch, maxlen, maxlen)
        mask = mask & mask.permute(0, 2, 1)

        return mask.unsqueeze(1)

    def forward_only_embedding_layer(self, token):
        """
        expect [batch, essay_idx_1] or [batch]
        return [seqlen, batch, embed_size]
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        embeddings = self.embedding_layer.forward(token.permute(1, 0))
        return embeddings

    def before_feed_to_decoder(self, last_step_output_token_embeddings, last_step_decoder_lstm_hidden,
                               last_step_decoder_lstm_memory, topics_representations):
        # mems [batch, mem_idx_per_sample]

        # calculate e_(y_{t-1})
        e_y_t_1 = last_step_output_token_embeddings

        # calculate c_{t}
        pre_t_matrix = self.W_1.forward(topics_representations)  # [batch, topic_num, temp]
        query_t = self.W_2.forward(last_step_decoder_lstm_memory).unsqueeze(2)
        # query_t [batch, 1, temp] for using add broadcast
        e_t_i = pre_t_matrix @ query_t  # [batch, topic_num, 1]
        alpha_t_i = torch.softmax(e_t_i, dim=1)  # [batch, topic_num, 1]
        c_t = topics_representations.permute(0, 2, 1) @ alpha_t_i

        # calculate m_{t}
        m_t = self.memory_neural.forward(last_step_decoder_lstm_hidden)

        return torch.cat([e_y_t_1.squeeze(), c_t.squeeze(), m_t.squeeze()], dim=1)

    def clear_memory_neural_step_state(self):
        self.memory_neural.step_mem_embeddings = None

    def init_memory_neural_step_state(self, begin_embeddings):
        self.memory_neural.step_mem_embeddings = begin_embeddings.permute(1, 0, 2)
        # step_mem_embeddings
        # reshape embeddings to [batch, len, embed_size]

    def forward(self, topic, topic_len, essay_input, mems, teacher_force_ratio=0.5):

        # topic_input [topic, topic_len]
        # topic [batch_size, seq_len]
        batch_size = topic.shape[0]
        max_essay_len = essay_input.shape[1]
        teacher_force_ratio = torch.tensor(teacher_force_ratio, dtype=torch.float, device=self.device)
        teacher_mode_chocie = torch.rand([max_essay_len], device=self.device)

        decoder_outputs = torch.zeros([batch_size, max_essay_len, self.vocab_size], device=self.device)

        topic_embeddings = self.forward_only_embedding_layer(topic).permute(1, 0, 2)
        mask = self.get_mask_matrix(topic)

        topics_representations, (h, c) = self.encoder.forward(topic_embeddings, mask)
        # [batch, topic_pad_num, output_size]

        mem_embeddings = self.forward_only_embedding_layer(mems)
        # [mem_max_num, batch, embed_size]
        self.init_memory_neural_step_state(mem_embeddings)

        # first input token is <go>
        # if lstm layer > 1, then select the topmost layer lstm memory c[-1] and hidden h[-1]
        now_input = essay_input[:, 0]
        now_input_embeddings = self.forward_only_embedding_layer(now_input)
        self.memory_neural.update_memory(now_input_embeddings)
        now_decoder_input = self.before_feed_to_decoder(now_input_embeddings, h[-1], c[-1],
                                                        topics_representations)

        for now_step in range(1, max_essay_len):
            logits, (h, c) = self.decoder.forward(now_decoder_input, h, c)
            decoder_outputs[:, now_step - 1] = logits
            if teacher_mode_chocie[now_step] < teacher_force_ratio:
                now_input = essay_input[:, now_step]
            else:
                now_input = logits.argmax(dim=-1)
            now_input_embeddings = self.forward_only_embedding_layer(now_input)
            self.memory_neural.update_memory(now_input_embeddings)
            now_decoder_input = self.before_feed_to_decoder(now_input_embeddings, h[-1], c[-1],
                                                            topics_representations)

        logits, _ = self.decoder.forward(now_decoder_input, h, c)
        decoder_outputs[:, -1] = logits
        self.clear_memory_neural_step_state()
        # [batch, essay_len, essay_vocab_size]
        return decoder_outputs


if __name__ == '__main__':


    pass
    batch = 64
    maxlen = 80
    dim = 100
    # print(outs.argmax(dim=2))
    idx = torch.randint(0, 100, [batch, maxlen])
    idxlen = torch.randint(1, maxlen, [batch])
    for i in range(batch):
        idx[i][idxlen[i]:] = 0
    print(idx)
    mask = get_mask_matrix(idx, 0)
    query = torch.rand([batch, maxlen, dim])
    key = torch.rand([batch, maxlen, dim])
    value = torch.rand([batch, maxlen, dim])
    model = MultiheadAttention(dim, 300, 6, dropout=0.5)
    res = model.forward(query, key, value, mask=mask.unsqueeze(1))




    pass