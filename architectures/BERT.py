import re
import math
import numpy as np
from random import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Embedding(nn.Module):
    """
    Embedding is the combination of:
                    - word embedding
                    - positional embedding
                    - segment embedding
    """
    def __init__(self, vocab_size, d_model, max_len, n_segments, dropout):
        super().__init__()

        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.seg_embdding = nn.Embedding(n_segments, d_model)

        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seg):
        # x => [batch_size, seq_len]
        # seg => [batch_size, seq_len]

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        positions = torch.arange(0, seq_len).unsqueeze(0)
        # positions => [1, seq_len]
        positions = positions.repeat(batch_size, 1)
        # positions => [batch_size, seq_len]

        embedding = self.word_embedding(x) + self.pos_embedding(positions) + self.seg_embdding(seg)
        embedding = self.dropout(self.ln_f(embedding))
        # embedding => [batch_size, seq_len, d_model]

        return embedding


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        assert d_model % n_heads == 0, "Number of heads must be a factor of d_model"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        # query, key, value => [batch_size, seq_len, d_model]

        batch_size = query.shape[0]


        Q, K, V = self.w_q(query), self.w_k(key), self.w_v(value)
        # Q, K, V => [batch_size, seq_len, d_model]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q, K, V => [batch_size, n_heads, seq_len, head_dim]

        # Q   => [batch_size, n_heads, seq_len, head_dim]
        # K^t => [batch_size, n_heads, head_dim, seq_len]
        # score => [batch_size, n_heads, seq_len, seq_len]
        score = torch.matmul(Q, K.transpose(-2, -1))
        # scaling the score
        score = score / torch.sqrt(torch.FloatTensor([self.head_dim]))

        if attn_mask:
            score = score.masked_fill(attn_mask == 0, -1e10)

        attention = torch.softmax(score, dim=-1)
        attention = self.dropout(attention)
        # attention => [batch_size, n_heads, seq_len, seq_len]

        # attention    => [batch_size, n_heads, seq_len, seq_len]
        # V            => [batch_size, n_heads, seq_len, head_dim]
        # weighted     => [batch_size, n_heads, seq_len, head_dim]
        weighted = torch.matmul(attention, V)

        weighted = weighted.permute(0, 2, 1, 3).contiguous()
        # weighted => [batch_size, seq_len, n_heads, head_dim]

        weighted = weighted.view(batch_size, -1, self.n_heads * self.head_dim)
        # weighted => [batch_size, seq_len, d_model]

        return weighted


class PositionWiseFeedForward(nn.Module):
    """
    Position wise feed forward network
    """
    def __init__(self, d_model, dff, dropout):
        super().__init__()

        self.pff = nn.Linear(d_model, dff)
        self.out = nn.Linear(dff, d_model)
        self.act_fn = nn.GELU()
        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x => [batch_size, seq_len, d_model]

        x = self.act_fn(self.pff(x))
        # x => [batch_size, seq_len, dff]

        x = self.out(x)
        x = self.dropout(self.ln_f(x))
        # x => [batch_size, seq_len, d_model]

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout):
        super().__init__()

        self.attn = Attention(d_model, n_heads, dropout)
        self.pff = PositionWiseFeedForward(d_model, dff, dropout)

        self.ln_f = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        # x => [batch_size, seq_len, d_model]

        # self-attention and layer normalization
        _x = self.attn(x, x, x, attn_mask)
        x = x + self.dropout(self.ln_f(_x))
        # x => [batch_size, seq_len, d_model]

        # position-wise feed forward and layer normalization
        _x = self.pff(x)
        x = x + self.dropout(self.ln_f(_x))
        # x => [batch_size, seq_len, d_model]

        return x


class BERT(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, dff, vocab_size, max_len, n_segments, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.embedding = Embedding(vocab_size, d_model, max_len, n_segments, dropout)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, dff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.nsp_linear = nn.Linear(d_model, d_model)
        self.nsp_classifier = nn.Linear(d_model, 2)

        self.mlm_linear = nn.Linear(d_model, d_model)
        self.mlm_act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)

        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        # share the embeddings weight with decoder
        self.decoder.weight = self.embedding.word_embedding.weight
        self.decoder_bias = nn.Parameter(torch.zeros([vocab_size]))

    def forward(self, input_ids, segment_ids, masked_pos):
        # input_ids => [batch_size, seq_len]
        # segment_ids => [batch_size, seq_len]
        # masked_pos => [batch_size, seq_len]

        encoded = self.embedding(input_ids, segment_ids)
        # encoded => [batch_size, seq_len, d_model]

        for layer in self.layers:
            encoded = layer(encoded)
        # encoded => [batch_size, seq_len, d_model]

        # Next Sentence Prediction (NSP)
        # for NSP the hidden state of first token [CLS] is taken and
        # sent through a prediction layer.
        hidden = encoded[:, 0, :]
        # hidden => [batch_size, d_model]

        nsp_hidden = torch.tanh(self.nsp_linear(hidden))
        nsp_hidden = self.dropout(nsp_hidden)
        # nsp_hidden => [batch_size, d_model]

        logits_clf = self.nsp_classifier(nsp_hidden)
        # logits_clf => [batch_size, 2]


        # Masked Language Model (MLM)
        masked_pos = masked_pos.unsqueeze(2)
        # masked_pos => [batch_size, seq_len, 1]

        # repeat the masked position d_model times
        masked_pos = masked_pos.repeat(1, 1, self.d_model)
        # masked_pos => [batch_size, seq_len, d_model]

        # we do not the complete inputs, only a part of them are masked.
        # MLM training objective is to predict the tokens at the masked inputs.
        # In order to do that, we have to get the encoded representation
        # at that particular masked inputs.
        # torch.gather will help in finding the values in the
        # input array (encoded) at the particular indices (masked_pos)
        # across a particular dimension (1 -> sequence dimension)
        masked_vals = torch.gather(encoded, 1, masked_pos)
        # masked_vals => [batch_size, seq_len, d_model]
        # output (masked_vals) is of same shape as masked_pos
        # for more information about torch.gather refer to the documentation
        # here: https://pytorch.org/docs/1.4.0/torch.html?highlight=gather#torch.gather

        # MLM layer propagation
        masked_hidden = self.mlm_linear(masked_vals)
        masked_hidden = self.mlm_act_fn(masked_hidden)
        masked_hidden = self.layer_norm(masked_hidden)
        # masked_hidden => [batch_size, seq_len, d_model]

        logits_mlm = self.decoder(masked_hidden) + self.decoder_bias
        # logits_mlm => [batch_size, seq_len, vocab_size]

        return logits_clf, logits_mlm



text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)

# The actual tokenization used in BERT is different
# WordPiece tokenizer was used in BERT
# Here for exploration purpose, simple space tokenization was used
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')   # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word_dict[w] = i + 4
number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)

token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)


max_len = 52
batch_size = 4
max_pred = 20   # max tokens of prediction
n_layers = 12
n_heads = 8
d_model = 64
dff = 64 * 4  # 4*d_model, FeedForward dimension
n_segments = 2
dropout = 0.1

model = BERT(
    d_model,
    n_layers,
    n_heads,
    dff,
    vocab_size,
    max_len,
    n_segments,
    dropout
)

# creating a batch of data for training
def make_batch():
    batch = []

    # next sentence
    positive = negative = 0

    # let's create a batch of data with same ratio
    while (positive != batch_size / 2) or (negative != batch_size / 2):
        # randomly pick sentence 1 and sentence 2
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))

        # get the tokens
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]

        # create the input to BERT by merging sentence 1 and sentence 2
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]

        # create the segment ids
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        #                   [CLS] + tokens_a + [SEP] + tokens_b + [SEP]

        # Mask LM
        # mask 15% of the tokens
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))

        cand_masked_pos = [i for i, token in enumerate(input_ids)]
        shuffle(cand_masked_pos)
        masked_tokens, masked_pos = [], []

        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])

            # 80% of the time use [MASK]
            if random() < 0.8:
                input_ids[pos] = word_dict['[MASK]']
            # 10% of the time replace with random
            elif random() < 0.5:
                index = randint(0, vocab_size - 1)
                input_ids[pos] = word_dict[number_dict[index]]
            # other 10% of the time keep the original token
        
        # padding
        n_pad = max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # next sentence prediction task
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1

    return batch


criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

batch = make_batch()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids = torch.LongTensor(input_ids)
segment_ids = torch.LongTensor(segment_ids)
masked_tokens = torch.LongTensor(masked_tokens)
masked_pos = torch.LongTensor(masked_pos)
isNext = torch.LongTensor(isNext)

model.train()
for epoch in range(25):
    optimizer.zero_grad()
    logits_clf, logits_mlm = model(input_ids, segment_ids, masked_pos)

    loss_clf = criterion1(logits_clf, isNext)
    loss_mlm = criterion2(logits_mlm.transpose(1, 2), masked_tokens)
    loss_mlm = (loss_mlm.float()).mean()

    loss = loss_clf + loss_mlm
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()
