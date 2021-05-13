import random

import torch
import torch.nn as nn
from tools import tools_load_pickle_obj, tools_get_logger

class CNNDiscriminator(nn.Module):
    def __init__(self, label_num, vocab_size, embed_size, channel_nums:list,
                 kernel_sizes:list):
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

class simple_seq2seq(nn.Module):
    def __init__(self, layer_num, hidden_size, vocab_size, embed_size, device):
        super(simple_seq2seq, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        # self.embedding_layer.requires_grad_()
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.device = device
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout()

    def forward_test(self, essay_input, essay_len, topic):
        batch_size = essay_input.shape[0]
        essay_pad_len = essay_input.shape[1]
        max_essay_len = torch.max(essay_len).item()
        decoder_outputs = torch.zeros([batch_size, essay_pad_len, self.vocab_size], device=self.device)
        # topic_embeddings = self.embedding_layer(topic.permute(1, 0))
        topic_embeddings = self.embedding_layer(topic.permute(1, 0))
        essay_input_embeddings = self.embedding_layer(essay_input.permute(1, 0))
        # [seq_len, batch, embed_size]
        outs, (h, c) = self.encoder.forward(topic_embeddings)

        # h, c = torch.rand([1, batch_size, self.hidden_size], device=self.device), torch.rand([1, batch_size, self.hidden_size], device=self.device)
        now_input = essay_input_embeddings[0, :].unsqueeze(0)

        now_step = 0
        for now_step in range(1, max_essay_len):
            now_input = self.dropout(now_input)
            outs, (h, c) = self.decoder(now_input, (h, c))

            temp = self.fc(outs.squeeze())

            decoder_outputs[:, now_step - 1] = temp
            now_input = decoder_outputs[:, now_step - 1].argmax(dim=-1).unsqueeze(0)
            now_input = self.embedding_layer(now_input)

        outs, (h, c) = self.decoder(now_input, (h, c))
        decoder_outputs[:, now_step] = self.fc(outs.squeeze())

        return decoder_outputs

    def forward_train(self, essay_input, essay_len, topic):
        # teacher_force_rate = torch.tensor(inputs[-1], dtype=torch.float, device=self.device)
        batch_size = essay_input.shape[0]
        essay_pad_len = essay_input.shape[1]
        # teacher_choice = torch.rand([essay_pad_len], device=self.device)
        max_essay_len = torch.max(essay_len).item()
        decoder_outputs = torch.zeros([batch_size, essay_pad_len, self.vocab_size], device=self.device)
        # topic_embeddings = self.embedding_layer(topic.permute(1, 0))
        topic_embeddings = self.embedding_layer(topic.permute(1, 0))
        essay_input_embeddings = self.embedding_layer(essay_input).permute(1, 0, 2)
        # [seq_len, batch, embed_size]
        outs, (h, c) = self.encoder.forward(topic_embeddings)
        # h, c = torch.rand([1, batch_size, self.hidden_size], device=self.device), torch.rand([1, batch_size, self.hidden_size], device=self.device)

        now_input = essay_input_embeddings[0, :].unsqueeze(0)
        now_step = 0
        for now_step in range(1, max_essay_len):
            now_input = self.dropout(now_input)
            outs, (h, c) = self.decoder(now_input, (h, c))
            decoder_outputs[:, now_step - 1] = self.fc(outs.squeeze())
            now_input = essay_input_embeddings[now_step, :].unsqueeze(0)

        now_step += 1
        outs, (h, c) = self.decoder(now_input, (h, c))
        decoder_outputs[:, now_step] = self.fc(outs.squeeze())

        return decoder_outputs

    def forward(self, topic, topic_len, essay_input, essay_len, mems, teacher_force_ratio=0.5):
        # topic, topic_len, essay_input, mems
        if teacher_force_ratio < 0.01:
            return self.forward_test(essay_input, essay_len, topic)
        else:
            return self.forward_train(essay_input, essay_len, topic)

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
        h = h.view(self.layer_num, self.direction, h.size(1), -1) #[layer, direction, batch, -1]
        c = c.index_select(1, idx_reverse)
        c = c.view(self.layer_num, self.direction, c.size(1), -1)

        h = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)
        c = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)

        # select the real final state
        # outs = outs[inputs[1]-1, torch.arange(outs.size(1)), :]

        return outs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, layer_num, encoder_output_size):
        super(Decoder, self).__init__()
        self.input_size = embed_size + encoder_output_size + embed_size # decoder_embed + encoder_out + memory_embed
        temp = self.input_size // 2
        self.sharp_linear = nn.Linear(self.input_size, temp)
        self.layer_num = layer_num
        self.embed_size = embed_size
        self.hidden_size = encoder_output_size
        self.input_size = temp
        self.lstm = nn.LSTM(self.input_size, encoder_output_size, layer_num,
                            bidirectional=False, dropout=0.5 if layer_num > 1 else 0.0)
        self.fc = nn.Linear(encoder_output_size, vocab_size)
        self.dropout = nn.Dropout(0.5)


    def forward(self, input_to_lstm, init_h, init_c):
        input_to_lstm = input_to_lstm.unsqueeze(0) # [1-single-token, batch, input_size]
        input_to_lstm = self.sharp_linear.forward(input_to_lstm)
        outs, (h, c) = self.lstm(input_to_lstm, (init_h, init_c))

        logits = self.fc(outs.squeeze(0))

        return self.dropout(logits), (h, c)

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
            decoder_embeddings = decoder_embeddings.permute(1, 0, 2) # [batch, 1, embed_size]
        # batch_size = self.step_mem_embeddings.shape[0]
        # seq_len = self.step_mem_embeddings.shape[1]

        M_t_temp = self.U1.forward(self.step_mem_embeddings.detach()) + self.V1.forward(decoder_embeddings) # broadcast add
        M_t_temp = torch.tanh(M_t_temp) # [batch, seq_len, embed_size]

        gate = self.U2.forward(self.step_mem_embeddings.detach()) + self.V2.forward(decoder_embeddings) # broadcast add
        gate = torch.sigmoid(gate) # [batch, seq_len, embed_size]

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

class KnowledgeEnhancedSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, encoder_lstm_hidden, encoder_bid,
                lstm_layer, attention_size, device):
        super(KnowledgeEnhancedSeq2Seq, self).__init__()
        self.pretrained_wv_path = pretrained_wv_path
        self.encoder = Encoder(embed_size, lstm_layer, encoder_lstm_hidden, encoder_bid)
        self.decoder = Decoder(vocab_size, embed_size, lstm_layer, self.encoder.output_size)
        self.memory_neural = Memory_neural(embed_size, self.decoder.hidden_size)
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.W_1 = nn.Linear(self.encoder.output_size, attention_size, bias=False)
        self.W_2 = nn.Linear(self.decoder.hidden_size, attention_size, bias=False)
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
        return embeddings

    def before_feed_to_decoder(self, last_step_output_token_embeddings, last_step_decoder_lstm_hidden,
                               last_step_decoder_lstm_memory, topics_representations):
        # mems [batch, mem_idx_per_sample]

        # calculate e_(y_{t-1})
        e_y_t_1 = last_step_output_token_embeddings

        # calculate c_{t}
        pre_t_matrix = self.W_1.forward(topics_representations) # [batch, topic_num, temp]
        query_t = self.W_2.forward(last_step_decoder_lstm_memory).unsqueeze(2)
        # query_t [batch, 1, temp] for using add broadcast
        e_t_i = pre_t_matrix @ query_t #[batch, topic_num, 1]
        alpha_t_i = torch.softmax(e_t_i, dim=1) # [batch, topic_num, 1]
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

    def forward(self, topic, topic_len, essay_input, essay_len, mems, teacher_force_ratio=0.5):

        # topic_input [topic, topic_len]
        # topic [batch_size, seq_len]
        batch_size = topic.shape[0]
        essay_pad_len = essay_input.shape[1]
        max_essay_len = torch.max(essay_len).item()
        decoder_outputs = torch.zeros([batch_size, essay_pad_len, self.vocab_size], device=self.device)

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

        now_step = 0
        for now_step in range(1, max_essay_len):
            logits, (h, c) = self.decoder.forward(now_decoder_input, h, c)
            decoder_outputs[:, now_step - 1] = logits
            if random.random() < teacher_force_ratio:
                now_input = essay_input[:, now_step]
            else:
                now_input = logits.argmax(dim=-1)
            now_input_embeddings = self.forward_only_embedding_layer(now_input)
            self.memory_neural.update_memory(now_input_embeddings)
            now_decoder_input = self.before_feed_to_decoder(now_input_embeddings, h[-1], c[-1],
                                                            topics_representations)
        logits, _ = self.decoder.forward(now_decoder_input, h, c)
        decoder_outputs[:, now_step] = logits
        self.clear_memory_neural_step_state()
        # [batch, essay_len, essay_vocab_size]
        return decoder_outputs

def init_param(self, init_way=None):
    if hasattr(self, 'init_parmas'):
        self.init_parmas()
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

def truncated_normal_(tensor, mean=0, std=1):
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




