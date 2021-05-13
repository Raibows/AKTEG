import torch
import random
import torch.nn as nn




class Encoder_LSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, is_bid):
        super(Encoder_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=is_bid, batch_first=True)
        self.direction = 2 if is_bid else 1
        self.dropout = nn.Dropout(0.5)
        self.h_c_size = hidden_size

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

        outs, _ = nn.utils.rnn.pad_packed_sequence(outs, padding_value=1e-31, batch_first=True)
        #
        outs = outs.index_select(0, idx_reverse)
        outs_fw, outs_bw = outs.split(outs.size(-1) // 2, dim=-1)
        outs = outs_fw + outs_bw
        h, c = h.index_select(1, idx_reverse), c.index_select(1, idx_reverse)
        h, c = h.view(1, self.direction, h.size(1), -1), c.view(1, self.direction, c.size(1), -1)# [layer, direction, batch, -1]
        h, c = h[:, 0, :, :] + h[:, 1, :, :], c[:, 0, :, :] + c[:, 1, :, :]
        # h, c = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1), torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)
        # outs [batch, seqlen, -1]
        # h or c [layernum, batch, -1]

        return outs, (h, c)


class AMLSTM(nn.Module):
    def __init__(self, hidden_size, embed_size, attention_size):
        super(AMLSTM, self).__init__()
        self.hidden_proj1 = nn.Linear(hidden_size, embed_size)
        self.hidden_proj2 = nn.Linear(hidden_size, attention_size)
        self.input_proj1 = nn.Linear(embed_size, embed_size)
        self.memory_proj = nn.Linear(embed_size, embed_size)
        self.encoder_outputs_proj = nn.Linear(hidden_size, attention_size)
        self.attention_v = nn.Parameter(torch.randn(attention_size))
        self.embed_size = embed_size
        self.memory = None
        self.mem_num = 120

    def init_memory(self, memory):
        self.memory = memory.detach()

    def cleal_memory(self):
        self.memory = None

    def forward(self, encoder_outputs, decoder_h, decoder_c, inputs, eps = 1e-40):
        """
        :param inputs: [batch_size x 1 x embedding_size]
        :param hidden: (h, c), [1, batch_size, hidden_size]
        :return:
        """
        # todo detach op of self.memory_embeding is for not updating the memory_corpus embedding weight through gradient

        # todo step calculate process m_t
        v = torch.tanh(self.hidden_proj1(decoder_h)).transpose(1, 0) # [batch_size, 1, embedding_size]
        score = v @ self.memory.transpose(1, 2)  # [batch_size, 1, mem_num]
        similarity = torch.softmax(score.squeeze(dim=1), dim=-1)    # [batch_size, mem_num]
        similarity = similarity.unsqueeze(dim=-1).repeat(1, 1, self.embed_size) # [batch_size, mem_num, embedding_size]
        mt = torch.sum(similarity * self.memory, dim=1)  # [batch_size, embedding_size]

        # todo update mem
        inputs_expand = inputs.repeat(1, self.mem_num, 1)  # [batch_size, mem_num, embedding_size]
        candidate = self.input_proj1(inputs_expand) + self.memory_proj(self.memory)  # [batch_size, mem_num, embedding_size]
        gate = self.memory @ inputs.transpose(1, 2)  # [batch_size, mem_num, 1]
        gate = torch.sigmoid(gate.squeeze(dim=-1))  # [batch_size, mem_num]
        gate = gate.unsqueeze(dim=-1).repeat(1, 1, self.embed_size) # [batch_size, mem_num, embedding_size]
        self.memory = (1. - gate) * self.memory + gate * candidate
        self.memory[self.memory.isinf()] = eps


        # todo c_t calculate
        encoder_processed = self.encoder_outputs_proj(encoder_outputs) # [batch_size, topic_num, attention_size]
        query_processed = self.hidden_proj2(decoder_c).transpose(0, 1)  # [batch_size, 1, attention_size]
        score = torch.sum(self.attention_v * torch.tanh(encoder_processed + query_processed), dim=-1)
        similarity = torch.softmax(score, dim=-1)   # [batch_size, topic_num]
        output_hidden_size = encoder_outputs.size(-1)
        similarity = similarity.unsqueeze(dim=-1).repeat(1, 1, output_hidden_size)  # [batch_size, topic_num, output_hidden_size]
        attn_value = torch.sum(encoder_outputs * similarity, dim=1)    # [batch_size, output_hidden_size]

        # todo generator cat process
        lstm_inp = torch.cat((inputs.squeeze(dim=1), attn_value, mt), dim=-1).unsqueeze(dim=1) # [batch_size, 1, lstm_input_size]

        return lstm_inp




class CTEG_official(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device, pretrained_wv_path):
        super(CTEG_official, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder_LSTM(embed_size, hidden_size, is_bid=True)
        self.memory = AMLSTM(hidden_size, embed_size, attention_size=128)
        self.input_size = hidden_size + embed_size + embed_size
        self.decoder = nn.LSTM(self.input_size, hidden_size, batch_first=True, num_layers=1)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.device = device
        self.dropout = nn.Dropout(0.5)
        self.pretrained_wv_path = pretrained_wv_path
        self.vocab_size = vocab_size


    def forward(self, topic, topic_len, essay_input, essay_len, mems, teacher_force_ratio=0.5):
        max_essay_len = torch.max(essay_len).item()
        batch, essay_pad_len = essay_input.shape
        topic_embed = self.embedding_layer.forward(topic) #[batch, topiclen, embed]
        encoder_outs, (h, c) = self.encoder.forward(topic_embed, topic_len)
        mem_embed = self.embedding_layer.forward(mems)
        decoder_outputs = torch.zeros([batch, essay_pad_len, self.vocab_size],
                                      device=self.device)
        self.memory.init_memory(mem_embed.detach())
        now_input = essay_input[:, 0].unsqueeze(1) #[batch, 1]

        i = 0
        for i in range(1, max_essay_len):
            now_input = self.embedding_layer.forward(now_input)
            decoder_input = self.memory.forward(encoder_outs, h, c, now_input)
            outs, (h, c) = self.decoder.forward(decoder_input, (h, c))
            outs = self.dropout(outs.squeeze())
            logits = self.fc_out.forward(outs)
            decoder_outputs[:, i - 1, :] = logits

            if random.random() < teacher_force_ratio:
                now_input = essay_input[:, i]
            else:
                now_input = logits.argmax(dim=1)
            now_input = now_input.unsqueeze(1)

        now_input = self.embedding_layer.forward(now_input)
        decoder_input = self.memory.forward(encoder_outs, h, c, now_input)
        outs, (h, c) = self.decoder.forward(decoder_input, (h, c))
        outs = self.dropout(outs.squeeze())
        logits = self.fc_out.forward(outs)
        decoder_outputs[:, i, :] = logits

        return decoder_outputs
