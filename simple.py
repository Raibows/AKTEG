import random
import torch
import torch.nn as nn
from utils import Encoder_LSTM

class simple_seq2seq(nn.Module):
    def __init__(self, layer_num, hidden_size, vocab_size, embed_size, device, pretrained_wv_path):
        super(simple_seq2seq, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder_LSTM(embed_size, layer_num, hidden_size, is_bid=True, is_cat=False, batch_first=True)
        self.decoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.device = device
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.5)
        self.pretrained_wv_path = pretrained_wv_path

    def forward_test(self, essay_input, topic, topic_len):
        batch_size = essay_input.shape[0]
        essay_pad_len = essay_input.shape[1]
        decoder_outputs = torch.zeros([batch_size, essay_pad_len, self.vocab_size], device=self.device)
        topic_embeddings = self.embedding_layer(topic)
        # [batch, seqlen, embed_size]
        _, (h, c) = self.encoder.forward(topic_embeddings, topic_len)

        sos_pos = essay_input[0, 0]
        logits = torch.zeros([batch_size, self.vocab_size], dtype=torch.float, device=self.device)
        logits[torch.arange(batch_size), sos_pos] = 1.0
        for i in range(essay_pad_len):
            now_input = logits.argmax(dim=-1).unsqueeze(1)
            now_input = self.embedding_layer.forward(now_input)
            outs, (h, c) = self.decoder.forward(now_input, (h, c))
            logits = self.fc(self.dropout(outs.squeeze()))
            decoder_outputs[:, i, :] = logits
        return decoder_outputs

    def forward_train(self, essay_input, topic, topic_len):
        batch_size = essay_input.shape[0]
        essay_pad_len = essay_input.shape[1]
        decoder_outputs = torch.zeros([batch_size, essay_pad_len, self.vocab_size], device=self.device)
        topic_embeddings = self.embedding_layer(topic)
        outs, (h, c) = self.encoder.forward(topic_embeddings, topic_len)

        essay_input_embeddings = self.embedding_layer(essay_input)
        for now_step in range(essay_pad_len):
            now_input = essay_input_embeddings[:, now_step, :].unsqueeze(1)
            outs, (h, c) = self.decoder(now_input, (h, c))
            decoder_outputs[:, now_step, :] = self.fc(self.dropout(outs.squeeze()))

        return decoder_outputs

    def forward(self, topic, topic_len, essay_input, essay_len, mems, teacher_force=True):
        # topic, topic_len, essay_input, mems
        if not teacher_force:
            return self.forward_test(essay_input, topic, topic_len)
        else:
            return self.forward_train(essay_input, topic, topic_len)


