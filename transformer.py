import torch
from torch import nn
from neural import Memory_neural
from magic import Attention
import random

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
        head_dim = self.output_dim // self.n_heads
        Q = self.W_q(query).reshape(batch, self.n_heads, -1, head_dim)
        K = self.W_k(key).reshape(batch, self.n_heads, -1, head_dim)
        V = self.W_v(value).reshape(batch, self.n_heads, -1, head_dim)

        attention = Q @ K.permute(0, 1, 3, 2) / self.scale #[batch, nhead, seq_len, seq_len]
        if mask != None:
            attention.masked_fill(mask == False, -1e10)

        attention = self.dropout(torch.softmax(attention, dim=-1))

        res = attention @ V #[batch, nhead, seqlen, temp]

        outs = res.permute(0, 2, 1, 3).contiguous().view(batch, -1, self.output_dim) # concat nheads

        outs = self.fc(outs)

        return outs, attention

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, query_dim, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(query_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, nheads, pf_dim, device):
        super(EncoderLayer, self).__init__()
        self.self_attn_norm_layer = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiheadAttention(hid_dim, hid_dim, nheads, dropout=0.5, device=device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout=0.5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention.forward(src, src, src, mask=src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_norm_layer(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src

class Encoder(nn.Module):
    def __init__(self, hid_dim, nlayers, nheads, pf_dim, topic_pad_num, device):
        super().__init__()
        self.device = device
        # self.pos_embedding = nn.Embedding(topic_pad_num, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, nheads, pf_dim, device) for _ in range(nlayers)
        ])
        self.scale = torch.sqrt(torch.tensor([hid_dim], dtype=torch.float, device=device))
        self.dropout = nn.Dropout(0.5)

    def forward(self, topic_embeddings, src_mask):
        batch, topic_len, _ = topic_embeddings.shape
        # pos = torch.arange(0, topic_len).unsqueeze(0).repeat(batch, 1).to(self.device)
        src = topic_embeddings * self.scale
        # src = topic_embeddings * self.scale + self.dropout(self.pos_embedding.forward(pos))
        for layer in self.layers:
            src = layer.forward(src, src_mask)
        #[batch, len, hidden]
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, nheads, pf_dim, device):
        super(DecoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiheadAttention(hid_dim, hid_dim, nheads, 0.5, device)
        self.encoder_attention = MultiheadAttention(hid_dim, hid_dim, nheads, 0.5, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, 0.5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, trg_embeddings, enc_src, trg_mask, src_mask):
        # trg_embeddings = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]
        _trg, _ = self.self_attention.forward(trg_embeddings, trg_embeddings, trg_embeddings, trg_mask)
        trg = self.self_attn_layer_norm(trg_embeddings + self.dropout(_trg))
        _trg, attention = self.encoder_attention.forward(trg_embeddings, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward.forward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention

class Decoder(nn.Module):
    def __init__(self, hid_dim, nlayers, nheads, pf_dim, max_essay_len, device):
        super(Decoder, self).__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_essay_len, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, nheads, pf_dim, device) for _ in range(nlayers)
        ])
        self.scale = torch.sqrt(torch.tensor([hid_dim], dtype=torch.float, device=device))
        self.dropout = nn.Dropout(0.5)

    def forward(self, essay_embeddings, enc_src, trg_mask, src_mask):
        # essay_embeddings = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]
        batch, trg_len, _ = essay_embeddings.shape
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch, 1).to(self.device)
        trg = essay_embeddings * self.scale + self.dropout(self.pos_embedding.forward(pos))
        for layer in self.layers:
            trg, attention = layer.forward(trg, enc_src, trg_mask, src_mask)

         #[batch, trglen, embed_size]
        return trg, attention

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, topic_pad_num, essay_pad_len, device, mask_idx):
        super(TransformerSeq2Seq, self).__init__()
        self.pretrained_wv_path = pretrained_wv_path
        self.encoder = Encoder(embed_size, 3, 4, 400, topic_pad_num, device)
        self.decoder = Decoder(embed_size, 3, 4, 400, essay_pad_len, device)
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.embed_size = embed_size
        self.mask_idx = torch.tensor(mask_idx, dtype=torch.int64, device=device)
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = nn.Dropout(0.5)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        src_mask = (src != self.mask_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.mask_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward_only_embedding_layer(self, token):
        """
        expect [batch, essay_idx_1] or [batch]
        return [seqlen, batch, embed_size]
        """
        if token.dim() == 1:
            token = token.unsqueeze(1)
        embeddings = self.embedding_layer.forward(token)
        return self.dropout(embeddings)

    def forward(self, topic, topic_len, essay_input, mems, teacher_force_ratio=0.5):
        batch = topic.shape[0]
        topic_mask = self.make_src_mask(topic)
        topic_embeddings = self.forward_only_embedding_layer(topic)
        enc_src = self.encoder.forward(topic_embeddings, topic_mask)

        if abs(teacher_force_ratio) < 1e-6: #test mode
            now_idx = essay_input[:, 0].unsqueeze(1)
            essay_maxlen = essay_input.shape[1]
            outputs = torch.zeros([batch, essay_maxlen, self.embed_size], device=self.device)
            for i in range(essay_maxlen):
                essay_mask = self.make_trg_mask(now_idx)
                essay_embeddings = self.forward_only_embedding_layer(now_idx)
                outs, attention = self.decoder.forward(essay_embeddings, enc_src, essay_mask, topic_mask)
                outs = outs.squeeze()
                outputs[:, i, :] = outs
                now_idx = outs.argmax(dim=-1).unsqueeze(1)
            return self.fc_out(outputs)
        else: # train mode
            essay_mask = self.make_trg_mask(essay_input)
            essay_embeddings = self.forward_only_embedding_layer(essay_input)
            outs, attention = self.decoder.forward(essay_embeddings, enc_src, essay_mask, topic_mask)
            return self.fc_out(outs)

class Knowledge(nn.Module):
    def __init__(self, embed_size, device):
        super(Knowledge, self).__init__()
        self.attention_layer = MultiheadAttention(embed_size, embed_size, 4, 0.5, device)
        self.enc_attention = Attention(embed_size, embed_size)
        self.self_attention_layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.5)
        self.step_embeddings = None

    def forward(self, essay_token_embeddings, topic_enc_outs, topic_mask, essay_token_mask):
        # essay_token_embeddings [batch, 1, embed_size]
        # topic_enc_outs [batch, topic_num, embed_size]
        energy = self.enc_attention.forward(topic_enc_outs, essay_token_embeddings.permute(1, 0, 2), topic_mask) #[batch, topic_num, 1]
        attention = (topic_enc_outs.permute(0, 2, 1) @ energy).permute(0, 2, 1) # [batch, 1, embed_size]
        query = self.self_attention_layer_norm(essay_token_embeddings + self.dropout(attention)) #[batch, 1, embed_size]
        outs, _ = self.attention_layer.forward(query, self.step_embeddings, self.step_embeddings, essay_token_mask)

        return outs

    def clear_step_embeddings(self):
        self.step_embeddings = None

    def init_step_embeddings(self, mem_embeddings):
        # [batch, max_mem_num, embed_size]
        self.step_embeddings = mem_embeddings

class KnowledgeTransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, topic_pad_num, essay_pad_len, device, mask_idx):
        super(KnowledgeTransformerSeq2Seq, self).__init__()
        self.encoder = Encoder(embed_size, 1, 4, 400, topic_pad_num, device)
        self.feed_to_decoder_size = embed_size + embed_size
        self.decoder = nn.GRU(self.feed_to_decoder_size, embed_size, num_layers=1, batch_first=True)
        self.knowledge = Knowledge(embed_size, device)
        self.fc_decoder_h = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
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

    def before_feed_to_decoder(self, now_input, enc_src, topic_mask):
        now_mask = self.make_trg_mask(now_input)
        now_embed = self.forward_only_embedding_layer(now_input)
        memory = self.knowledge.forward(now_embed, enc_src, topic_mask, now_mask)
        feeds = torch.cat([now_embed, memory], dim=2)  # [batch, 1, 2*embed_size]

        return feeds

    def forward(self, topic, topic_len, essay_input, mems, teacher_force_ratio=0.5):
        batch, essay_len = essay_input.shape
        topic_mask = self.make_src_mask(topic)
        topic_embeddings = self.forward_only_embedding_layer(topic)
        enc_src = self.encoder.forward(topic_embeddings, topic_mask) #[batch, len, embed_size]
        mem_embeddings = self.forward_only_embedding_layer(mems)
        self.knowledge.init_step_embeddings(mem_embeddings.detach())
        decoder_outputs = torch.zeros([batch, essay_len, self.vocab_size], device=self.device)
        now_input = essay_input[:, 0].unsqueeze(1) #[batch, 1]
        h = torch.sum(enc_src, dim=1) / topic_len.unsqueeze(1)
        h = torch.tanh(self.fc_decoder_h(h)).unsqueeze(0) #[1, batch, embed_size]

        for i in range(1, essay_len):
            feeds = self.before_feed_to_decoder(now_input, enc_src, topic_mask)
            outs, h = self.decoder.forward(feeds, h)
            outs = outs.squeeze()
            logits = self.fc_out(outs) #[batch, vocab_size]
            decoder_outputs[:, i-1, :] = logits

            if random.random() < teacher_force_ratio:
                now_input = essay_input[:, i]
            else:
                now_input = logits.argmax(dim=1)
            now_input = now_input.unsqueeze(1)

        feeds = self.before_feed_to_decoder(now_input, enc_src, topic_mask)
        outs, _ = self.decoder.forward(feeds, h)
        outs = outs.squeeze()
        logits = self.fc_out(outs)
        decoder_outputs[:, -1, :] = logits
        self.knowledge.clear_step_embeddings()
        return decoder_outputs

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.mask_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.mask_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

class Knowledge_v2(nn.Module):
    def __init__(self, embed_size, decoder_hidden_size, device):
        super(Knowledge_v2, self).__init__()
        self.attention_layer = MultiheadAttention(embed_size, 512, 4, 0.5, device)
        self.enc_attention = Attention(embed_size, decoder_hidden_size)
        self.self_attention_layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(0.5)
        self.step_embeddings = None

    def forward(self, essay_token_embeddings, decoder_hidden, topic_enc_outs, topic_mask, essay_token_mask):
        # essay_token_embeddings [batch, 1, embed_size]
        # topic_enc_outs [batch, topic_num, embed_size]
        energy = self.enc_attention.forward(topic_enc_outs, decoder_hidden, topic_mask) #[batch, topic_num, 1]
        attention = (topic_enc_outs.permute(0, 2, 1) @ energy).permute(0, 2, 1) # [batch, 1, embed_size]
        query = self.self_attention_layer_norm(essay_token_embeddings + self.dropout(attention)) #[batch, 1, embed_size]
        outs, _ = self.attention_layer.forward(query, self.step_embeddings, self.step_embeddings, essay_token_mask)

        feeds = torch.cat([essay_token_embeddings, outs, attention], dim=2)

        return feeds

    def clear_step_embeddings(self):
        self.step_embeddings = None

    def init_step_embeddings(self, mem_embeddings):
        # [batch, max_mem_num, embed_size]
        self.step_embeddings = mem_embeddings

class KnowledgeTransformerSeq2Seqv2(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, topic_pad_num, essay_pad_len, device, mask_idx):
        super(KnowledgeTransformerSeq2Seqv2, self).__init__()
        decoder_hidden_size = 512
        self.encoder = Encoder(embed_size, 1, 4, 512, topic_pad_num, device)
        self.feed_to_decoder_size = embed_size + 512 + embed_size
        self.decoder = nn.GRU(self.feed_to_decoder_size, decoder_hidden_size, num_layers=1, batch_first=True)
        self.knowledge = Knowledge_v2(embed_size, decoder_hidden_size, device)
        self.fc_decoder_h = nn.Linear(embed_size, decoder_hidden_size)
        self.fc_out = nn.Linear(decoder_hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
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

    def before_feed_to_decoder(self, now_input, last_hidden, enc_src, topic_mask):
        now_mask = self.make_trg_mask(now_input)
        now_embed = self.forward_only_embedding_layer(now_input)
        feeds = self.knowledge.forward(now_embed, last_hidden, enc_src, topic_mask, now_mask)

        return feeds

    def forward(self, topic, topic_len, essay_input, mems, teacher_force_ratio=0.5):
        batch, essay_len = essay_input.shape
        topic_mask = self.make_src_mask(topic)
        topic_embeddings = self.forward_only_embedding_layer(topic)
        enc_src = self.encoder.forward(topic_embeddings, topic_mask) #[batch, len, embed_size]
        mem_embeddings = self.forward_only_embedding_layer(mems)
        self.knowledge.init_step_embeddings(mem_embeddings.detach())
        decoder_outputs = torch.zeros([batch, essay_len, self.vocab_size], device=self.device)
        now_input = essay_input[:, 0].unsqueeze(1) #[batch, 1]
        h = torch.sum(enc_src, dim=1) / topic_len.unsqueeze(1)
        h = torch.tanh(self.fc_decoder_h(h)).unsqueeze(0) #[1, batch, embed_size]

        for i in range(1, essay_len):
            feeds = self.before_feed_to_decoder(now_input, h, enc_src, topic_mask)
            outs, h = self.decoder.forward(feeds, h)
            outs = outs.squeeze()
            logits = self.fc_out(outs) #[batch, vocab_size]
            decoder_outputs[:, i-1, :] = logits

            if random.random() < teacher_force_ratio:
                now_input = essay_input[:, i]
            else:
                now_input = logits.argmax(dim=1)
            now_input = now_input.unsqueeze(1)

        feeds = self.before_feed_to_decoder(now_input, h, enc_src, topic_mask)
        outs, _ = self.decoder.forward(feeds, h)
        outs = outs.squeeze()
        logits = self.fc_out(outs)
        decoder_outputs[:, -1, :] = logits
        self.knowledge.clear_step_embeddings()
        return decoder_outputs

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.mask_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.mask_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

class Knowledge_v3(nn.Module):
    def __init__(self, embed_size, enc_out_size, device):
        super(Knowledge_v3, self).__init__()
        self.enc_attention = Attention(enc_out_size, embed_size)
        # self.mem_attention = Attention(embed_size, enc_out_size)
        self.mem_attention = MultiHeadAttentionLayer(enc_out_size, embed_size, 4, 0.5, device)
        self.step_embeddings = None

    def forward(self, dec_embedings, topic_enc_outs, topic_mask, mem_mask):
        # dec_embedings [batch, 1, embed_size]
        # topic_enc_outs [batch, topic_num, embed_size]
        energy_t = self.enc_attention.forward(topic_enc_outs, dec_embedings, topic_mask) #[batch, topic_len, 1]
        attn_t = topic_enc_outs.permute(0, 2, 1) @ energy_t #[batch, enc_out_size, 1]
        attn_mem, _ = self.mem_attention.forward(attn_t.permute(0, 2, 1), self.step_embeddings, self.step_embeddings, mask=mem_mask)
        # energy_mem = self.mem_attention.forward(self.step_embeddings, attn_t)
        # attn_mem = self.step_embeddings.permute(0, 2, 1) @ energy_mem

        feeds = torch.cat([dec_embedings, attn_mem], dim=-1)

        return feeds # [batch, 1, embed_size + embed_size]


    def clear_step_embeddings(self):
        self.step_embeddings = None

    def init_step_embeddings(self, mem_embeddings):
        # [batch, max_mem_num, embed_size]
        self.step_embeddings = mem_embeddings.detach()

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
        # [batch

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
        # todo fw + bw
        outs = outs_fw + outs_bw
        h, c = h.index_select(1, idx_reverse), c.index_select(1, idx_reverse)
        h, c = h.view(1, self.direction, h.size(1), -1), c.view(1, self.direction, c.size(1), -1)# [layer, direction, batch, -1]
        h, c = h[:, 0, :, :] + h[:, 1, :, :], c[:, 0, :, :] + c[:, 1, :, :]
        # h, c = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1), torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)

        # select the real final state
        # outs = outs[inputs[1]-1, torch.arange(outs.size(1)), :]

        return outs, (h, c)

class KnowledgeTransformerSeq2Seqv3(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained_wv_path, topic_pad_num, essay_pad_len, device, mask_idx):
        super(KnowledgeTransformerSeq2Seqv3, self).__init__()
        self.encoder = Encoder_LSTM(embed_size, 512, is_bid=True)
        self.decoder_hidden_size = self.encoder.h_c_size
        self.feed_to_decoder_size = embed_size + embed_size
        self.decoder = nn.LSTM(self.feed_to_decoder_size, self.decoder_hidden_size, num_layers=1, batch_first=True)
        self.knowledge = Knowledge_v3(embed_size, self.encoder.h_c_size, device)
        self.fc_out = nn.Linear(self.decoder_hidden_size, vocab_size)
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

    def before_feed_to_decoder(self, now_input, enc_outs, topic_mask, mem_mask):
        now_embed = self.forward_only_embedding_layer(now_input)
        feeds = self.knowledge.forward(now_embed, enc_outs, topic_mask, mem_mask)

        return feeds

    def forward(self, topic, topic_len, essay_input, mems, teacher_force_ratio=0.5):
        batch, essay_len = essay_input.shape
        topic_mask = self.make_src_mask(topic)
        topic_embeddings = self.forward_only_embedding_layer(topic)
        enc_outs, (h, c) = self.encoder.forward(topic_embeddings, topic_len) #[batch, len, embed_size]
        mem_embeddings = self.forward_only_embedding_layer(mems)
        mem_mask = self.make_src_mask(mems)
        self.knowledge.init_step_embeddings(mem_embeddings.detach())
        decoder_outputs = torch.zeros([batch, essay_len, self.vocab_size], device=self.device)
        now_input = essay_input[:, 0].unsqueeze(1) #[batch, 1]


        for i in range(1, essay_len):
            feeds = self.before_feed_to_decoder(now_input, enc_outs, topic_mask, mem_mask)
            outs, (h, c) = self.decoder.forward(feeds, (h, c))
            outs = outs.squeeze()
            logits = self.fc_out(outs) #[batch, vocab_size]
            decoder_outputs[:, i-1, :] = logits

            if random.random() < teacher_force_ratio:
                now_input = essay_input[:, i]
            else:
                now_input = logits.argmax(dim=1)
            now_input = now_input.unsqueeze(1)

        feeds = self.before_feed_to_decoder(now_input, enc_outs, topic_mask, mem_mask)
        outs, _ = self.decoder.forward(feeds, (h, c))
        outs = outs.squeeze()
        logits = self.fc_out(outs)
        decoder_outputs[:, -1, :] = logits
        self.knowledge.clear_step_embeddings()
        return decoder_outputs

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.mask_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.mask_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

if __name__ == '__main__':


    # pass
    # batch = 64
    # maxlen = 80
    # dim = 100
    # # print(outs.argmax(dim=2))
    # idx = torch.randint(0, 100, [batch, maxlen])
    # idxlen = torch.randint(1, maxlen, [batch])
    # for i in range(batch):
    #     idx[i][idxlen[i]:] = 0
    # print(idx)
    # mask = get_mask_matrix(idx, 0)
    # query = torch.rand([batch, maxlen, dim])
    # key = torch.rand([batch, maxlen, dim])
    # value = torch.rand([batch, maxlen, dim])
    # model = MultiheadAttention(dim, 300, 6, dropout=0.5)
    # res = model.forward(query, key, value, mask=mask.unsqueeze(1))




    pass
