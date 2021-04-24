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
        self.lstm = nn.LSTM(embed_size, hidden_size, layer_num, bidirectional=is_bid,
                            dropout=0.5 if layer_num > 1 else 0.0)
        self.direction = 2 if is_bid else 1
        self.output_size = self.direction * self.hidden_size
        self.dropout = nn.Dropout(0.5)

    def forward(self, *inputs):
        # (sen, sen_len)
        embeddings = self.embedding_layer(inputs[0].permute(1, 0))
        embeddings = self.dropout(embeddings)

        sort = torch.sort(inputs[1], descending=True)
        sent_len_sort, idx_sort = sort.values, sort.indices
        idx_reverse = torch.argsort(idx_sort)

        sent = embeddings.index_select(1, idx_sort)

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sort.cpu())
        outs, (h, c) = self.lstm(sent_packed)
        outs, _ = nn.utils.rnn.pad_packed_sequence(outs, padding_value=-1e9)

        outs = outs.index_select(1, idx_reverse)
        h = h.index_select(1, idx_reverse)
        h = h.reshape(self.layer_num, h.size(1), -1)
        c = c.index_select(1, idx_reverse)
        c = c.reshape(self.layer_num, c.size(1), -1)

        # select the real final state
        # outs = outs[inputs[1]-1, torch.arange(outs.size(1)), :]

        return outs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, encoder_output_size, memory_neural_embed_size, pretrained_path):
        super(Decoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.input_size = embed_size + encoder_output_size + memory_neural_embed_size
        if pretrained_path:
            self.embedding_layer.from_pretrained(
                torch.tensor(tools_load_pickle_obj(pretrained_path), dtype=torch.float)
            )
        self.embedding_layer.weight.requires_grad = True
        self.layer_num = layer_num
        self.embed_size = embed_size
        self.hidden_size = encoder_output_size
        self.lstm = nn.LSTM(self.input_size, encoder_output_size, layer_num,
                            bidirectional=False, dropout=0.5 if layer_num > 1 else 0.0)
        self.fc = nn.Linear(encoder_output_size, vocab_size)
        self.dropout = nn.Dropout(0.5)


    def forward(self, input_to_lstm, init_h, init_c):
        input_to_lstm = input_to_lstm.unsqueeze(0) # [1-single-token, batch, input_size]
        outs, (h, c) = self.lstm(input_to_lstm, (init_h, init_c))

        logits = self.fc(outs.squeeze(0))
        logits = self.dropout(logits)

        return logits, (h, c)

class Memory_neural(nn.Module):
    def __init__(self, vocab_size, embed_size, decoder_hidden_size, decoder_embed_size, pretrained_path, embedding_grad):
        super(Memory_neural, self).__init__()
        self.embed_size = embed_size
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        if not pretrained_path:
            raise AssertionError(f'memory neural must have pretrained word vectors bug got {pretrained_path}')

        self.embedding_layer.from_pretrained(
            torch.tensor(tools_load_pickle_obj(pretrained_path), dtype=torch.float)
        )
        # using gate mechanism is for step-by-step update
        # still needs grad descent
        self.embedding_layer.weight.requires_grad = embedding_grad
        self.W = nn.Linear(decoder_hidden_size, embed_size, bias=True)
        self.U1 = nn.Linear(embed_size, embed_size)
        self.V1 = nn.Linear(decoder_embed_size, embed_size)
        self.U2 = nn.Linear(embed_size, embed_size)
        self.V2 = nn.Linear(decoder_embed_size, embed_size)
        self.step_mem_embeddings = None
        self.dropout = nn.Dropout(0.5)

    def update_memory(self, decoder_embeddings):
        """
        note decoder_embeddings is t_step but not (t-1) last step
        step_mem_embeddings [batch, seq_len, embed_size]
        decoder_embeddings [batch, 1, embed_size]
        """
        decoder_embeddings = decoder_embeddings.squeeze()
        batch_size = self.step_mem_embeddings.shape[0]
        seq_len = self.step_mem_embeddings.shape[1]
        M_t_temp = self.dropout(self.U1(self.step_mem_embeddings.reshape(-1, self.embed_size)).reshape(batch_size, seq_len, -1)) \
                   + self.dropout(self.V1(decoder_embeddings).reshape(batch_size, 1, self.embed_size))
        M_t_temp = torch.tanh(M_t_temp) # [batch, seq_len, embed_size]

        gate = self.dropout(self.U2(self.step_mem_embeddings.reshape(-1, self.embed_size)).reshape(batch_size, seq_len, -1)) \
                   + self.dropout(self.V2(decoder_embeddings).reshape(batch_size, 1, self.embed_size))
        gate = torch.sigmoid(gate) # [batch, seq_len, embed_size]

        self.step_mem_embeddings = M_t_temp * gate + self.step_mem_embeddings * (1 - gate)
        self.step_mem_embeddings = self.dropout(self.step_mem_embeddings)


    def forward(self, decoder_hidden_s_t_1, mems):
        if self.step_mem_embeddings == None:
            mems = mems.permute(1, 0)
            embeddings = self.embedding_layer(mems).permute(1, 0, 2)
            embeddings = self.dropout(embeddings)
            self.step_mem_embeddings = embeddings
            # embeddings [batch, len, embed_size]


        v_t = torch.tanh(self.dropout(self.W(decoder_hidden_s_t_1)))
        # v_t here is a column vector [batch, v_t] using torch.batch_multiplication
        q_t = torch.softmax(self.dropout(self.step_mem_embeddings @ v_t.unsqueeze(2)), dim=1)
        # q_t here is a column vector [batch, q_t]
        m_t = q_t.permute(0, 2, 1) @ self.step_mem_embeddings

        return m_t

class Seq2Seq(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, memory_neural:Memory_neural, topic_padding_num, essay_vocab_size, attention_size, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.memory_neural = memory_neural
        self.essay_vocab_size = essay_vocab_size
        self.device = device
        self.dropout = nn.Dropout(0.5)
        self.W_1 = nn.Linear(encoder.output_size, attention_size)
        self.W_2 = nn.Linear(self.decoder.hidden_size, attention_size)
        self.W_3 = nn.Linear(topic_padding_num, topic_padding_num)

    def forward_only_decoder_embedding_layer(self, token):
        """
        expect [batch, essay_idx1] or [batch]
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        embeddings = self.decoder.embedding_layer.forward(token.permute(1, 0))
        return self.dropout(embeddings)

    def before_feed_to_decoder(self, last_step_output_token_embeddings, last_step_decoder_lstm_hidden,
                               last_step_decoder_lstm_memory, topics_representations, mems):
        # mems [batch, mem_idx_per_sample]

        # calculate e_(y_{t-1})
        e_y_t_1 = last_step_output_token_embeddings

        # calculate c_{t}
        batch_size = topics_representations.shape[1]
        topic_num = topics_representations.shape[0]
        pre_t_matrix = self.dropout(self.W_1.forward(topics_representations.reshape(batch_size * topic_num, -1)))
        pre_t_matrix = pre_t_matrix.reshape(batch_size, topic_num, -1)
        query_t = self.dropout(self.W_2.forward(last_step_decoder_lstm_memory).unsqueeze(2))
        # query_t [batch, t, 1] for using torch.batch_multiplication
        e_t_i = self.dropout(self.W_3.forward(torch.tanh(pre_t_matrix @ query_t).squeeze(2)))
        alpha_t_i = torch.softmax(e_t_i, dim=1) # [batch, topic_num]
        c_t = topics_representations.reshape(batch_size, -1, topic_num) @ alpha_t_i.unsqueeze(2)

        # calculate m_{t}
        m_t = self.memory_neural.forward(last_step_decoder_lstm_hidden, mems)

        return torch.cat([e_y_t_1.squeeze(), c_t.squeeze(), m_t.squeeze()], dim=1)

    def clear_memory_neural_step_state(self):
        self.memory_neural.step_mem_embeddings = None

    def forward(self, topic_len_input:tuple, essay_input:torch.Tensor, mems:torch.Tensor, teacher_force_ratio=0.5):

        # topic_input [topic, topic_len]
        # topic [batch_size, seq_len]
        teacher_force_ratio = torch.tensor(teacher_force_ratio, dtype=torch.float, device=self.device)
        batch_size = topic_len_input[0].shape[0]
        max_essay_len = essay_input.shape[1]
        teacher_mode_chocie = torch.rand([max_essay_len], device=self.device)


        decoder_outputs = torch.zeros([max_essay_len, batch_size, self.essay_vocab_size], device=self.device)

        topics_representations, (h, c) = self.encoder(topic_len_input[0], topic_len_input[1]) #[topic_pad_num, batch, output_size]

        # first input token is <sos>
        # if lstm layer > 1, then select the topmost layer lstm memory c[-1] and hidden h[-1]
        now_input = essay_input[:, 0]
        now_input_embeddings = self.forward_only_decoder_embedding_layer(now_input)
        now_decoder_input = self.before_feed_to_decoder(now_input_embeddings, h[-1], c[-1], topics_representations, mems)

        for now_step in range(1, max_essay_len):
            logits, (h, c) = self.decoder.forward(now_decoder_input, h, c)
            decoder_outputs[now_step - 1] = logits
            if teacher_mode_chocie[now_step] < teacher_force_ratio:
                now_input = essay_input[:, now_step]
            else:
                now_input = torch.multinomial(torch.softmax(logits, dim=1), num_samples=1)
                # now_input = logits.argmax(1)
            now_input_embeddings = self.forward_only_decoder_embedding_layer(now_input)
            self.memory_neural.update_memory(now_input_embeddings)
            now_decoder_input = self.before_feed_to_decoder(now_input_embeddings, h[-1], c[-1], topics_representations, mems)

        logits, _ = self.decoder.forward(now_decoder_input, h, c)
        decoder_outputs[-1] = logits
        self.clear_memory_neural_step_state()
        # [essay_len, batch, essay_vocab_size]
        return decoder_outputs

def uniform_init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

if __name__ == '__main__':




    pass