import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import youtokentome

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, pad_idx, device):
        super().__init__()

        assert d_model % n_heads == 0, "n_heads must be a factor of d_model"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.fc = nn.Linear(d_model, d_model)

        self.pad_idx = pad_idx
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        # query => [batch_size, seq_len, d_model] 
        # key => [batch_size, seq_len, d_model]
        # value => [batch_size, seq_len, d_model]

        batch_size = query.shape[0]

        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)
        # Q, K, V => [batch_size, seq_len, d_model]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q, K, V => [batch_size, n_heads, seq_len, head_dim]

        energy = torch.matmul(Q, K.permute(0 ,1, 3, 2))
        energy = energy / self.scale
        # energy => [batch_size, n_heads, query_len, key_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        # attention => [batch_size, n_heads, query_len, key_len]

        weighted = torch.matmul(attention, V)
        # weighted => [batch_size, n_heads, query_len, head_dim]

        weighted = weighted.permute(0, 2, 1, 3).contiguous()
        # weighted => [batch_size, query_len, n_heads, head_dim]

        x = weighted.view(batch_size, -1, self.d_model)
        # x => [batch_size, query_len, d_model]

        x = self.fc(x)
        # x => [batch_size, query_len, d_model]
        # attention => [batch_size, n_heads, query_len, key_len]

        return x, attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, pff_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(d_model, pff_dim)
        self.fc2 = nn.Linear(pff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        # input => [batch_size, seq_len, d_model]

        x = self.dropout(torch.relu(self.fc1(input)))
        # x => [batch_size, seq_len, pff_dim]

        x = self.fc2(x)
        # x => [batch_size, seq_len, d_model]

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, pff_dim, dropout, pad_idx, device):
        super().__init__()

        self.self_attention = SelfAttention(d_model, n_heads, dropout, pad_idx, device)
        self.pff = PositionWiseFeedForward(d_model, pff_dim, dropout)

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.pff_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src => [batch_size, src_len, d_model]

        # self attention on src
        _src, _ = self.self_attention(src, src, src, src_mask)
        # _src => [batch_size, src_len, d_model]

        # residual connection and layer normalization
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src => [batch_size, src_len, d_model]

        # position wise feed forward
        _src = self.pff(src)
        # _src => [batch_size, src_len, d_model]

        # residual connection and layer normalization
        src = self.pff_layer_norm(src + self.dropout(_src))
        # src => [batch_size, src_len, d_model]

        return src


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, pff_dim, dropout, pad_idx, device, max_len=500):
        super().__init__()

        self.n_layers = n_layers
        self.device = device
        self.word_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, pff_dim, dropout, pad_idx, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, src, src_mask=None):
        # src => [batch_size, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos => [batch_size, src_len]

        word_embed = self.word_embedding(src)
        word_embed = word_embed * self.scale
        # word_embed => [batch_size, src_len, d_model]

        pos_embed = self.pos_embedding(pos)
        # pos_embed => [batch_size, src_len, d_model]

        src = self.dropout(word_embed + pos_embed)
        # src => [batch_size, src_len, d_model]

        for layer in self.layers:
            src = layer(src, src_mask)
        
        # src => [batch_size, src_len, d_model]
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, pff_dim, dropout, pad_idx, device):
        super().__init__()

        self.self_attention = SelfAttention(d_model, n_heads, dropout, pad_idx, device)
        self.enc_attention = SelfAttention(d_model, n_heads, dropout, pad_idx, device)
        self.pff = PositionWiseFeedForward(d_model, pff_dim, dropout)

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model)
        self.pff_layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # _trg => [batch_size, trg_len, d_model]

        # residual connection and layer normalization
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg => [batch_size, trg_len, d_model]

        # enc_attention
        _trg, attention = self.enc_attention(trg, enc_src, enc_src, src_mask)
        # _trg => [batch_size, trg_len, d_model]
        # attention => [batch_size, n_heads, trg_len, src_len]

        # residual connection and layer normalization
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg => [batch_size, trg_len, d_model]

        # positionwise feed forward
        _trg = self.pff(trg)
        # _trg => [batch_size, trg_len, d_model]

        # residual connection and layer normalization
        trg = self.pff_layer_norm(trg + self.dropout(_trg))
        # trg => [batch_size, trg_len, d_model]

        return trg, attention


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, pff_dim, dropout, pad_idx, device, max_len=500):
        super().__init__()
        
        self.device = device
        self.word_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, pff_dim, dropout, pad_idx, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(d_model, output_dim)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, trg, enc_src, trg_mask=None, src_mask=None):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos => [batch_size, trg_len]

        word_embedding = self.word_embedding(trg)
        word_embedding = word_embedding * self.scale
        # word_embedding => [batch_size, trg_len, d_model]

        pos_embedding = self.pos_embedding(pos)
        # pos_embedding => [batch_size, trg_len, d_model]

        trg = self.dropout(word_embedding + pos_embedding)
        # trg => [batch_size, trg_len, d_model]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        logits = self.fc_out(trg)
        # logits => [batch_size, trg_len, output_dim]
        # attention => [batch_size, n_heads, trg_len, src_len]

        return logits, attention


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        # src_mask => [batch_size, 1, 1, src_len]
        
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2).to(self.device)
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask => [batch_size, 1, trg_len, trg_len]

        return trg_mask
    
    def forward(self, src, trg):
        # src => [batch_size, src_len]
        # trg => [batch_size, trg_len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention


bpe_model = youtokentome.BPE(model=os.path.join("model", "bpe.model"))

PAD_IDX = 0
INPUT_DIM = bpe_model.vocab_size()
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT,
              PAD_IDX,
              device)

dec = Decoder(INPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              PAD_IDX,
              device)

model = Transformer(enc, dec, PAD_IDX, device).to(device)