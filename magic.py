import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_size, layer_num, hidden_size, is_bid):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lstm = nn.LSTM(embed_size, hidden_size, layer_num, bidirectional=is_bid,
                            dropout=0.5 if layer_num > 1 else 0.0)
        self.direction = 2 if is_bid else 1
        self.output_size = self.direction * self.hidden_size

    def forward(self, *inputs):
        # (sen_embeddings, sen_len)

        sort = torch.sort(inputs[1], descending=True)
        sent_len_sort, idx_sort = sort.values, sort.indices
        idx_reverse = torch.argsort(idx_sort)

        sent = inputs[0].index_select(1, idx_sort)

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sort.cpu())
        # outs, (h, c) = self.lstm(inputs[0])
        outs, (h, c) = self.lstm.forward(sent_packed)

        outs, _ = nn.utils.rnn.pad_packed_sequence(outs, padding_value=1e-31)
        #
        outs = outs.index_select(1, idx_reverse)
        h = h.index_select(1, idx_reverse)
        h = h.view(self.layer_num, self.direction, h.size(1), -1)  # [layer, direction, batch, -1]
        c = c.index_select(1, idx_reverse)
        c = c.view(self.layer_num, self.direction, c.size(1), -1)

        h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)
        c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)
        h, c = torch.tanh(h), torch.tanh(c)

        # select the real final state
        # outs = outs[inputs[1]-1, torch.arange(outs.size(1)), :]

        return outs, (h, c)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, encoder_output_size):
        super(Decoder, self).__init__()
        self.input_size = embed_size + encoder_output_size + embed_size  # decoder_embed + encoder_out + memory_embed
        self.layer_num = layer_num
        self.embed_size = embed_size
        self.hidden_size = encoder_output_size
        self.lstm = nn.LSTM(self.input_size, encoder_output_size, layer_num,
                            bidirectional=False, dropout=0.5 if layer_num > 1 else 0.0)
        self.fc_out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, input_to_lstm, init_h, init_c):
        input_to_lstm = input_to_lstm.unsqueeze(0)  # [1-single-token, batch, input_size]
        outs, (h, c) = self.lstm(input_to_lstm, (init_h, init_c))
        logits = self.fc_out(outs.squeeze(0))
        return logits, (h, c)


class Memory_neural(nn.Module):
    def __init__(self, embed_size, decoder_hidden_size):
        super(Memory_neural, self).__init__()

        # using gate mechanism is for step-by-step update
        # still needs grad descent

        self.W = nn.Linear(decoder_hidden_size, embed_size, bias=True)
        self.U1 = nn.Linear(embed_size, embed_size, bias=False)
        self.V1 = nn.Linear(embed_size, embed_size, bias=False)
        self.U2 = nn.Linear(embed_size, embed_size, bias=False)
        self.V2 = nn.Linear(embed_size, embed_size, bias=False)
        self.step_mem_embeddings = None
        self.embed_size = embed_size

    def update_memory(self, decoder_embeddings):
        """
        note decoder_embeddings is t_step but not (t-1) last step
        step_mem_embeddings [batch, seq_len, embed_size]
        decoder_embeddings [batch, 1, embed_size] (convert to)
        """
        if decoder_embeddings.dim() == 2:
            decoder_embeddings = decoder_embeddings.unsqueeze(1)
        else:
            decoder_embeddings = decoder_embeddings.permute(1, 0, 2)  # [batch, 1, embed_size]
        # batch_size = self.step_mem_embeddings.shape[0]
        # seq_len = self.step_mem_embeddings.shape[1]

        M_t_temp = self.U1.forward(self.step_mem_embeddings.detach()) + self.V1.forward(
            decoder_embeddings)  # broadcast add
        M_t_temp = torch.tanh(M_t_temp)  # [batch, seq_len, embed_size]

        gate = self.U2.forward(self.step_mem_embeddings.detach()) + self.V2.forward(decoder_embeddings)  # broadcast add
        gate = torch.sigmoid(gate)  # [batch, seq_len, embed_size]

        self.step_mem_embeddings = gate * M_t_temp + (1.0 - gate) * self.step_mem_embeddings.detach()
        self.step_mem_embeddings[self.step_mem_embeddings.isinf()] = 1e-31
        self.step_mem_embeddings[self.step_mem_embeddings.isnan()] = 1e-31

    def forward(self, decoder_hidden_s_t_1):
        # self.step_mem_embeddings
        # reshape embeddings to [batch, len, embed_size]

        v_t = torch.tanh(self.W(decoder_hidden_s_t_1)).unsqueeze(2)
        # v_t here is a column vector [batch, v_t] using torch.batch_multiplication
        q_t = torch.softmax(self.step_mem_embeddings.detach() @ v_t, dim=1)
        # q_t here is a column vector [batch, q_t]
        m_t = q_t.permute(0, 2, 1) @ self.step_mem_embeddings.detach()

        return m_t


class Attention(nn.Module):
    def __init__(self, enc_output_size, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_output_size + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_ouputs, dec_hidden, enc_mask=None):
        # encoder_outputs [batch, topic_num, enc_output_size]
        # dec_hidden [1, batch, dec_dim]
        batch, topic_num, _ = encoder_ouputs.shape
        dec_hidden = dec_hidden.squeeze(0).unsqueeze(1).repeat(1, topic_num, 1)  # [batch, topic_num, dec_dim]
        energy = torch.cat([encoder_ouputs, dec_hidden], dim=2)
        energy = torch.tanh(self.attn.forward(energy))
        attention = self.v.forward(energy)  # [batch, topic_num, 1]
        if enc_mask != None:
            attention = attention.masked_fill(enc_mask.squeeze().unsqueeze(2) == False, -1e10)

        return torch.softmax(attention, dim=1)


class MagicSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, encoder_lstm_hidden, encoder_bid,
                 lstm_layer, device):
        super(MagicSeq2Seq, self).__init__()
        self.pretrained_wv_path = pretrained_wv_path
        self.encoder = Encoder(embed_size, lstm_layer, encoder_lstm_hidden, encoder_bid)
        self.decoder = Decoder(vocab_size, embed_size, lstm_layer, self.encoder.output_size)
        self.memory_neural = Memory_neural(embed_size, self.decoder.hidden_size)
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.attention_layer = Attention(self.encoder.output_size, self.decoder.hidden_size)
        self.fc_out = nn.Linear(self.encoder.output_size+self.decoder.hidden_size+embed_size, vocab_size)
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = nn.Dropout(0.5)

    def forward_only_embedding_layer(self, token):
        """
        expect [batch, essay_idx_1] or [batch]
        return [seqlen, batch, embed_size]
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        embeddings = self.embedding_layer.forward(token.permute(1, 0))
        return self.dropout(embeddings)

    def before_feed_to_decoder(self, last_step_output_token_embeddings, last_step_decoder_lstm_hidden,
                               last_step_decoder_lstm_memory, topics_representations):
        # mems [batch, mem_idx_per_sample]
        # topic_representations [batch, topic_num, enc_output_size]

        # calculate e_(y_{t-1})
        e_y_t_1 = last_step_output_token_embeddings

        # calculate c_{t}
        attention = self.attention_layer.forward(topics_representations, last_step_decoder_lstm_hidden) # [batch, topic_num]
        c_t = topics_representations.permute(0, 2, 1) @ attention

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

        topic_embeddings = self.forward_only_embedding_layer(topic)
        topics_representations, (h, c) = self.encoder.forward(topic_embeddings, topic_len)
        topics_representations = topics_representations.permute(1, 0, 2)
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