import torch
import torch.nn as nn
from utils import truncated_normal_
from utils import Encoder_LSTM
from utils import Decoder_proj




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
        score = v @ self.memory.detach().transpose(1, 2)  # [batch_size, 1, mem_num]
        similarity = torch.softmax(score.squeeze(dim=1), dim=-1)    # [batch_size, mem_num]
        similarity = similarity.unsqueeze(dim=-1).repeat(1, 1, self.embed_size) # [batch_size, mem_num, embedding_size]
        mt = torch.sum(similarity * self.memory.detach(), dim=1)  # [batch_size, embedding_size]

        # todo update mem
        inputs_expand = inputs.repeat(1, self.mem_num, 1)  # [batch_size, mem_num, embedding_size]
        candidate = self.input_proj1(inputs_expand) + self.memory_proj(self.memory.detach())  # [batch_size, mem_num, embedding_size]
        gate = self.memory.detach() @ inputs.transpose(1, 2)  # [batch_size, mem_num, 1]
        gate = torch.sigmoid(gate.squeeze(dim=-1))  # [batch_size, mem_num]
        gate = gate.unsqueeze(dim=-1).repeat(1, 1, self.embed_size) # [batch_size, mem_num, embedding_size]
        self.memory = (1. - gate) * self.memory.detach() + gate * candidate
        self.memory[self.memory.isinf()] = eps
        self.memory[self.memory.isnan()] = eps


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
        self.encoder = Encoder_LSTM(embed_size, 1, hidden_size, is_bid=True, is_cat=False, batch_first=True)
        self.memory = AMLSTM(hidden_size, embed_size, attention_size=128)
        self.input_size = hidden_size + embed_size + embed_size
        self.decoder = nn.LSTM(self.input_size, hidden_size, batch_first=True, num_layers=1)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.device = device
        self.dropout = nn.Dropout(0.5)
        self.pretrained_wv_path = pretrained_wv_path
        self.vocab_size = vocab_size

    def init_params(self):
        import math
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                truncated_normal_(param, std=stddev)

        for param in self.memory.input_proj1.parameters():
            if param.requires_grad:
                truncated_normal_(param, std=0.05)
        for param in self.memory.memory_proj.parameters():
            if param.requires_grad:
                truncated_normal_(param, std=0.05)
        truncated_normal_(self.memory.attention_v, std=0.05)
        truncated_normal_(self.memory.hidden_proj1.weight, std=0.05)
        self.memory.hidden_proj1.bias.data.zero_()

    def forward(self, topic, topic_len, essay_input, essay_len, mems, teacher_force=True):
        batch, essay_pad_len = essay_input.shape
        topic_embed = self.embedding_layer.forward(topic) #[batch, topiclen, embed]
        encoder_outs, (h, c) = self.encoder.forward(topic_embed, topic_len)
        mem_embed = self.embedding_layer.forward(mems)
        decoder_outputs = torch.zeros([batch, essay_pad_len, self.vocab_size], device=self.device)
        self.memory.init_memory(mem_embed.detach())

        if teacher_force:
            essay_input_embeddings = self.embedding_layer.forward(essay_input)
            for i in range(essay_pad_len):
                now_input = essay_input_embeddings[:, i, :].unsqueeze(1)
                decoder_input = self.memory.forward(encoder_outs, h, c, now_input)
                outs, (h, c) = self.decoder.forward(decoder_input, (h, c))
                outs = self.dropout(outs.squeeze())
                logits = self.fc_out.forward(outs)
                decoder_outputs[:, i, :] = logits
        else:
            sos_pos = essay_input[0, 0]
            logits = torch.zeros([batch, self.vocab_size], dtype=torch.float, device=self.device)
            logits[torch.arange(batch), sos_pos] = 1.0
            for i in range(essay_pad_len):
                now_input = logits.argmax(dim=1).unsqueeze(1)
                now_input = self.embedding_layer.forward(now_input)
                decoder_input = self.memory.forward(encoder_outs, h, c, now_input)
                outs, (h, c) = self.decoder.forward(decoder_input, (h, c))
                outs = self.dropout(outs.squeeze())
                logits = self.fc_out.forward(outs)
                decoder_outputs[:, i, :] = logits


        return decoder_outputs

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
        decoder_embeddings [batch, 1, embed_size]
        """

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

class CTEG_paper(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, encoder_lstm_hidden, encoder_bid,
                 lstm_layer, attention_size, device):
        super(CTEG_paper, self).__init__()
        self.pretrained_wv_path = pretrained_wv_path
        self.encoder = Encoder_LSTM(embed_size, lstm_layer, encoder_lstm_hidden, encoder_bid, is_cat=True, batch_first=True)
        self.decoder = Decoder_proj(vocab_size, embed_size, lstm_layer, self.encoder.output_size)
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
        return [batch, seqlen, embed_size]
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        embeddings = self.embedding_layer.forward(token)
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
        self.memory_neural.step_mem_embeddings = begin_embeddings
        # step_mem_embeddings
        # reshape embeddings to [batch, len, embed_size]

    def forward(self, topic, topic_len, essay_input, essay_len, mems, teacher_force=True):

        # topic_input [topic, topic_len]
        # topic [batch_size, seq_len]
        batch_size = topic.shape[0]
        essay_pad_len = essay_input.shape[1]
        decoder_outputs = torch.zeros([batch_size, essay_pad_len, self.vocab_size], device=self.device)

        topic_embeddings = self.forward_only_embedding_layer(topic)
        topics_representations, (h, c) = self.encoder.forward(topic_embeddings, topic_len)
        topics_representations = topics_representations
        # [batch, topic_pad_num, output_size]

        mem_embeddings = self.forward_only_embedding_layer(mems)
        # [mem_max_num, batch, embed_size]
        self.init_memory_neural_step_state(mem_embeddings)

        # first input token is <go>
        # if lstm layer > 1, then select the topmost layer lstm memory c[-1] and hidden h[-1]

        if teacher_force:
            essay_input_embeddings = self.embedding_layer.forward(essay_input)
            for i in range(essay_pad_len):
                now_input_embeddings = essay_input_embeddings[:, 0, :].unsqueeze(1)
                self.memory_neural.update_memory(now_input_embeddings)
                now_decoder_input = self.before_feed_to_decoder(now_input_embeddings, h[-1], c[-1], topics_representations)
                logits, (h, c) = self.decoder.forward(now_decoder_input, h, c)
                decoder_outputs[:, i, :] = logits.squeeze()
        else:
            sos_pos = essay_input[0, 0]
            logits = torch.zeros([batch_size, self.vocab_size], dtype=torch.float, device=self.device)
            logits[torch.arange(batch_size), sos_pos] = 1.0
            for i in range(essay_pad_len):
                now_input = logits.argmax(dim=-1)
                now_input_embeddings = self.forward_only_embedding_layer(now_input)
                self.memory_neural.update_memory(now_input_embeddings)
                now_decoder_input = self.before_feed_to_decoder(now_input_embeddings, h[-1], c[-1], topics_representations)
                logits, (h, c) = self.decoder.forward(now_decoder_input, h, c)
                decoder_outputs[:, i, :] = logits.squeeze()

        self.clear_memory_neural_step_state()
        return decoder_outputs


