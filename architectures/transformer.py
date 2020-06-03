'''
This code contains the implementation of transformer model.
Reference code: https://github.com/bentrevett/pytorch-seq2seq
'''

"""
There are few important blocks present in Transformer:
    => Positional Encoding
    => Self Attention
        -> Encoder - Encoder attention
        -> Decoder - Decoder attention
        -> Encoder - Decoder attention
    => Residual Connections & Layer Normalization
    => Positionwise Feed Forward
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding.

    There are other ways of implementing positional encoding.
    Below mentioned is the way where the positional encoding is fixed.
    Other way is to make that as a learnable using: nn.Embedding(max_len, d_model)
    Paper claims there is not much of a difference between either ways.
    """
    def __init__(self, d_model, dropout, max_len=1000):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))

        # pe(pos, 2i) = sin(pos / (10000 ^ (2i/d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)

        # pe(pos, 2i+1) = cos(pos / (10000 ^ (2i/d_model)))
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # pe => [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x => [batch_size, seq_len, d_model]
        
        # get the positional embeddings of word and add it
        x = x + self.pe[:, :x.shape[1], :]
        # x => [batch_size, seq_len, d_model]

        return self.dropout(x)


class SelfAttention(nn.Module):
    """
    Implements the self-attention layer.

    This is the core of the transformer model.
    There are three kinds of self-attention in transformer:
        > Encoder - Encoder: Does the self-attention on the src sentence
        > Decoder - Decoder: Does the self-attention on the trg sentence
        > Encoder - Decoder: Attends to the src sentence while decoding a particular word
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "Number of attention heads must be a factor of d_model"
        # in paper d_model = 512, n_heads = 8

        # query, key, value weight matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.drop = nn.Dropout(dropout)

        # linear layer to be applied after concating the attention head outputs
        self.fc = nn.Linear(d_model, d_model)

        # scale factor to be applied in calculation of self-attention
        self.scale = torch.sqrt(torch.FloatTensor([d_model // n_heads]))

    def forward(self, query, key, value, mask=None):
        # query => [batch_size, seq_len, d_model]
        # key => [batch_size, seq_len, d_model]
        # value => [batch_size, seq_len, d_model]
        # mask => [batch_size, 1, seq_len(query), seq_len(key)]

        batch_size = query.shape[0]
        hid_dim = query.shape[2]
        assert hid_dim == self.d_model, "Hidden dimensions must match"

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V => [batch_size, seq_len, d_model]

        Q = Q.view(batch_size, -1, self.n_heads, hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q, K, V => [batch_size, n_heads, seq_len, head_dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy => [batch_size, n_heads, seq_len(query), seq_len(key)]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.drop(attention)
        # attention => [batch_size, n_heads, seq_len(query), seq_len(key)]

        weighted = torch.matmul(attention, V)
        # weighted => [batch_size, n_heads, seq_len(query), head_dim]

        weighted = weighted.permute(0, 2, 1, 3)
        # weighted => [batch_size, seq_len(query), n_heads, head_dim]

        weighted = weighted.contiguous()
        weighted = weighted.view(batch_size, -1, hid_dim)
        # weighted => [batch_size, seq_len(query), d_model]

        output = self.fc(weighted)
        # output => [batch_size, seq_len(query), d_model]

        return output, attention


class PositionWiseFeedForward(nn.Module):
    """
    Implements the Position wise feed forward layer.

    It is a simple linear layer that project the input to higher dimension
    and then back project to the dimension feasible to the model.
    """
    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input):
        # input => [batch_size, seq_len, d_model]

        x = self.dropout(torch.relu(self.fc1(input)))
        # x => [batch_size, seq_len, hidden_dim]

        out = self.fc2(x)
        # out => [batch_size, seq_len, d_model]

        return out


class EncoderLayer(nn.Module):
    """
    Implement the single encoder block.

    There are 6 encoder blocks in transformer (according to the paper).
    Each encoder block encapsulates self-attention, positionwise feedforward layer
    and then the residual connections across each layer.
    """
    def __init__(self, d_model, n_heads, pff_dim, dropout):
        super().__init__()

        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.pff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = SelfAttention(d_model, n_heads, dropout)
        self.pff = PositionWiseFeedForward(d_model, pff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        # src => [batch_size, seq_len, d_model]
        # src_mask => [batch_size, 1(n_heads), seq_len(query), seq_len(key)]

        # self_attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # residual connection with layer norm
        src = self.self_attention_layer_norm(src + self.dropout(_src))
        # src => [batch_size, seq_len, d_model]

        # positionwise feed forward
        _src = self.pff(src)

        # residual connection with layer norm
        src = self.pff_layer_norm(src + self.dropout(_src))
        # src => [batch_size, seq_len, d_model]

        return src


class Encoder(nn.Module):
    """
    Implements the encoder.

    It takes the input and applies the word embedding, position embedding
    for each word and then it is passed through the encoder blocks.
    """
    def __init__(self, input_dim, d_model, n_layers, n_heads, pff_dim, dropout):
        super().__init__()

        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, pff_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        # src => [batch_size, seq_len]
        # src_mask => [batch_size, 1, seq_len(query), seq_len(key)]

        word_embedding = self.tok_embedding(src)
        # word_embedding => [batch_size, seq_len, d_model]

        position_embedding = self.pos_embedding(word_embedding)

        src = self.dropout(position_embedding)

        for layer in self.layers:
            src = layer(src, src_mask)
            # src => [batch_size, seq_len, d_model]

        return src


class DecoderLayer(nn.Module):
    """
    Implement the single decoder block.

    There are 6 decoder blocks in transformer (according to the paper).
    Each decoder block encapsulates self-attention(decoder), self-attention(encoder-decoder),
    positionwise feedforward layer and then the residual connections across each layer.
    """
    def __init__(self, d_model, n_heads, pff_dim, dropout):
        super().__init__()

        self.self_attention = SelfAttention(d_model, n_heads, dropout)
        self.enc_attention = SelfAttention(d_model, n_heads, dropout)
        self.pff = PositionWiseFeedForward(d_model, pff_dim, dropout)
        self.self_attention_layer_norm = nn.LayerNorm(d_model)
        self.enc_attention_layer_norm = nn.LayerNorm(d_model)
        self.pff_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg => [batch_size, trg_len, d_model]
        # enc_src => [batch_size, src_len, d_model]
        # trg_mask => [batch_size, trg_len]
        # src_mask => [batch_size, src_len]

        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # _trg => [batch_size, trg_len, d_model]

        # residual connection and layer norm
        trg = self.self_attention_layer_norm(trg + self.dropout(_trg))
        # trg => [batch_size, trg_len, d_model]

        # encoder attention
        _trg, attention = self.self_attention(trg, enc_src, enc_src, src_mask)
        # _trg => [batch_size, trg_len, d_model]
        # attention => [batch_size, n_heads, trg_len, src_len]

        # residual connection and layer norm
        trg = self.enc_attention_layer_norm(trg + self.dropout(_trg))
        # trg => [batch_size, trg_len, d_model]

        # positionwise feed forward
        _trg = self.pff(trg)

        # residual connection and layer norm
        trg = self.pff_layer_norm(trg + self.dropout(_trg))
        # trg => [batch_size, trg_len, d_model]

        return trg, attention


class Decoder(nn.Module):
    """
    Implements the decoder.

    It takes the input and applies the word embedding, position embedding
    for each word and then it is passed through the decoder blocks. At the
    end, it is passed through the linear layer to predict the token.
    """
    def __init__(self, output_dim, d_model, n_layers, n_heads, pff_dim, dropout):
        super().__init__()

        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, pff_dim, dropout) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg => [batch_size, trg_len]
        # enc_src => [batch_size, src_len, d_model]
        # trg_mask => [batch_size, 1, trg_len(query), trg_len(key)]
        # src_mask => [batch_size, 1, 1(query), src_len(key)]

        word_embedding = self.tok_embedding(trg)
        # word_embedding => [batch_size, trg_len, d_model]

        position_embedding = self.pos_embedding(word_embedding)
        # position_embedding => [batch_size, trg_len, d_model]

        trg = self.dropout(position_embedding)
        # trg => [batch_size, trg_len, d_model]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            # trg => [batch_size, trg_len, d_model]
            # attention => [batch_size, n_heads, trg_len, src_len]

        logits = self.fc(trg)
        # logits => [batch_size, trg_len, output_dim]

        return logits, attention


class Transformer(nn.Module):
    """
    Implements the transformer.

    It encapsulates the encoder and decoder. Inputs to encoder & decoder are padded
    to make batch processing feasible. While performing attention certain parts of
    the input need not be attended. This will taken care by input masks. Also while
    decoding a word at a certain position, it can only attend to the inputs present
    to the left side of it, not the right(future) inputs. So the decoder input masking
    contains an extra step, for each input it masks the inputs right-side of it.
    """
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # src => [batch_size, src_len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask => [batch_size, 1, 1, src_len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg => [batch_size, trg_len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask => [batch_size, 1, 1, trg_len]

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).bool()
        # trg_sub_mask => [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask => [batch_size, 1, trg_len, trg_len]

        return trg_mask

    def forward(self, src, trg):
        # src => [batch_size, src_len]
        # trg => [batch_size, trg_len]

        src_mask = self.make_src_mask(src)
        # src_mask => [batch_size, 1, 1, src_len]

        trg_mask = self.make_trg_mask(trg)
        # trg_mask => [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)
        # enc_src => [batch_size, src_len, d_model]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output => [batch_size, trg_len, output_dim]
        # attention => [batch_size, n_heads, trg_len, src_len]

        return output, attention


d_model = 16
pff_dim = 32
n_heads = 4
n_layers = 2
input_dim = 20
output_dim = 15
dropout = 0.4
src_pad_idx = 0
trg_pad_idx = 0
encoder = Encoder(input_dim, d_model, n_layers, n_heads, pff_dim, dropout)
decoder = Decoder(output_dim, d_model, n_layers, n_heads, pff_dim, dropout)
model = Transformer(encoder, decoder, src_pad_idx, trg_pad_idx)

src = torch.LongTensor(torch.randint(0, 20, (6, 8)))
trg = torch.LongTensor(torch.randint(0, 15, (6, 8)))
print(model)
output, attn = model(src, trg)
print(output.shape)
print(attn.shape)
