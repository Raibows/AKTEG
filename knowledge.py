import torch
from torch import nn
from magic import Attention
from utils import Encoder_LSTM
from transformer import MultiHeadAttentionLayer

class Knowledge_v3(nn.Module):
    def __init__(self, embed_size, dec_hid_size, enc_out_size, device):
        super(Knowledge_v3, self).__init__()
        self.enc_attention = Attention(enc_out_size, dec_hid_size)
        # self.mem_attention = Attention(embed_size, enc_out_size)
        self.mem_attention = MultiHeadAttentionLayer(enc_out_size, embed_size, 4, 0.5, device)
        self.res_mem_attention = MultiHeadAttentionLayer(dec_hid_size, embed_size, 4, 0.5, device)
        self.attn_mem_layer_norm = nn.LayerNorm(embed_size)
        self.step_embeddings = None
        self.dropout = nn.Dropout(0.5)

    def forward(self, dec_hidden, topic_enc_outs, topic_mask, mem_mask):
        # dec_hidden [batch, 1, dec_hidden_size]
        # topic_enc_outs [batch, topic_num, embed_size]
        energy_t = self.enc_attention.forward(topic_enc_outs, dec_hidden, topic_mask) #[batch, topic_len, 1]
        attn_t = topic_enc_outs.permute(0, 2, 1) @ energy_t #[batch, enc_out_size, 1]
        attn_mem, _ = self.mem_attention.forward(attn_t.permute(0, 2, 1), self.step_embeddings, self.step_embeddings, mask=mem_mask)
        attn_res_mem, _ = self.res_mem_attention.forward(dec_hidden, self.step_embeddings, self.step_embeddings, mask=mem_mask)

        attn_mem = attn_res_mem + self.dropout(attn_mem)
        attn_mem = self.attn_mem_layer_norm.forward(attn_mem)


        return attn_mem # [batch, 1, embed_size + embed_size]

    def clear_step_embeddings(self):
        self.step_embeddings = None

    def init_step_embeddings(self, mem_embeddings):
        # [batch, max_mem_num, embed_size]
        self.step_embeddings = mem_embeddings.detach()

class KnowledgeSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, device, mask_idx):
        super(KnowledgeSeq2Seq, self).__init__()
        self.encoder = Encoder_LSTM(embed_size, layer_num=1, hidden_size=512,
                                    is_bid=True, is_cat=False, batch_first=True)
        self.decoder_hidden_size = self.encoder.output_size
        self.feed_to_decoder_size = embed_size + embed_size
        self.decoder = nn.LSTM(self.feed_to_decoder_size, self.decoder_hidden_size, num_layers=1, batch_first=True)
        self.knowledge = Knowledge_v3(embed_size, self.decoder_hidden_size, self.encoder.output_size, device)
        self.fc_cat = nn.Linear(self.decoder_hidden_size + embed_size, vocab_size)
        self.fc_hidden = nn.Linear(self.decoder_hidden_size, vocab_size)
        self.gate = nn.Linear(self.decoder_hidden_size+embed_size, vocab_size)
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.pretrained_wv_path = pretrained_wv_path
        self.mask_idx = mask_idx
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.device = device

    def forward_only_embedding_layer(self, token):
        """
        expect [batch, essay_idx_1] or [batch]
        return [seqlen, batch, embed_size]
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        embeddings = self.embedding_layer.forward(token)
        return self.dropout(embeddings)

    def before_feed_to_decoder(self, now_input, dec_hidden, enc_outs, topic_mask, mem_mask):
        # now_input [batch, 1]
        # dec_hidden [1, batch, dec_size]

        now_embed = self.forward_only_embedding_layer(now_input)
        attn_mem = self.knowledge.forward(dec_hidden.permute(1, 0, 2), enc_outs, topic_mask, mem_mask)

        feeds = torch.cat([now_embed, attn_mem], dim=2)

        return feeds, attn_mem.squeeze()

    def get_logits(self, decoder_outs, attn_mem):
        """
        :param decoder_outs: [batch, hidden_size]
        :param attn_mem: [batch, embed_size]
        :return: logits [batch, vocab_size]
        """
        cat = torch.cat([decoder_outs, attn_mem], dim=-1)
        logits_hidden = self.fc_hidden.forward(self.dropout(decoder_outs))
        logits_cat = self.fc_cat.forward(self.dropout(cat))
        gate = torch.sigmoid(self.gate.forward(cat))
        logits = gate * logits_cat + (1.0 - gate) * logits_hidden

        return logits

    def forward(self, topic, topic_len, essay_input, essay_len, mems, teacher_force=True):
        batch, essay_pad_len = essay_input.shape
        topic_mask = self.make_src_mask(topic)
        topic_embeddings = self.forward_only_embedding_layer(topic)
        enc_outs, (h, c) = self.encoder.forward(topic_embeddings, topic_len) #[batch, len, embed_size]
        mem_embeddings = self.forward_only_embedding_layer(mems)
        mem_mask = self.make_src_mask(mems)
        self.knowledge.init_step_embeddings(mem_embeddings.detach())
        decoder_outputs = torch.zeros([batch, essay_pad_len, self.vocab_size], device=self.device)

        if teacher_force:
            for i in range(essay_pad_len):
                now_input = essay_input[:, i].unsqueeze(1)
                feeds, attn_mem = self.before_feed_to_decoder(now_input, h, enc_outs,
                                                              topic_mask, mem_mask)
                outs, (h, c) = self.decoder.forward(feeds, (h, c))
                logits = self.get_logits(outs.squeeze(), attn_mem)
                decoder_outputs[:, i, :] = logits
        else:
            sos_pos = essay_input[0, 0]
            logits = torch.zeros([batch, self.vocab_size], dtype=torch.float, device=self.device)
            logits[torch.arange(batch), sos_pos] = 1.0
            for i in range(essay_pad_len):
                now_input = logits.argmax(dim=1).unsqueeze(1)
                feeds, attn_mem = self.before_feed_to_decoder(now_input, h, enc_outs,
                                                              topic_mask, mem_mask)
                outs, (h, c) = self.decoder.forward(feeds, (h, c))
                logits = self.get_logits(outs.squeeze(), attn_mem)
                decoder_outputs[:, i, :] = logits

        self.knowledge.clear_step_embeddings()
        return decoder_outputs

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.mask_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask