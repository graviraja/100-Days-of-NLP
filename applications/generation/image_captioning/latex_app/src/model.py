import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN = 0


class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding.
    There are other ways of implementing positional encoding.
    Below mentioned is the way where the positional encoding is fixed.
    Other way is to make that as a learnable using: nn.Embedding(max_len, d_model)
    Paper claims there is not much of a difference between either ways.
    """
    def __init__(self, d_model, dropout, max_len=512):
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



class Encoder(nn.Module):
    def __init__(self, enc_out_dim=512):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(1, 1),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, enc_out_dim, 3, 1, 1),
            nn.ReLU()
        )

        self.pos = PositionalEncoding(enc_out_dim, 0.3)

    def forward(self, imgs):
        # imgs => [batch, 3, 32, 128]

        features = self.cnn(imgs)
        # features => [batch, 512, 2, 8]

        features = features.permute(0, 2, 3, 1)
        batch, H, W, _ = features.shape
        encoded_imgs = features.contiguous().view(batch, H*W, -1)
        encoded_imgs = self.pos(encoded_imgs)
        return encoded_imgs


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)

        self.full_attn = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        # encoder_out => [batch_size, num_pixels, enc_dim]
        # decoder_hidden => [batch_size, dec_dim]

        enc_attn = self.encoder_attn(encoder_out)
        # enc_attn => [batch_size, num_pixels, attn_dim]

        dec_attn = self.decoder_attn(decoder_hidden)
        # dec_attn => [batch_size, attn_dim]

        attn = self.full_attn(self.relu(enc_attn + dec_attn.unsqueeze(1)))
        # attn => [batch_size, num_pixels, 1]

        attn = attn.squeeze(2)
        # attn => [batch_size, num_pixels]

        alpha = self.softmax(attn)

        weighted = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        # weighted => [batch_size, encoder_dim]

        return weighted, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, decoder_dim, attention_dim, encoder_dim=512, dropout=0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_TOKEN)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            decoder_dim,
            batch_first=True
        )

        self.out = nn.Linear(emb_dim + encoder_dim + decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, trg, hidden, cell, encoder_out):
        # trg => [batch_size, 1]

        embedded = self.embedding(trg)
        # embedded => [batch_size, 1, emb_dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output => [batch_size, 1, hid_dim]
        # hidden, cell => [num_layers, batch_size, hid_dim]

        weighted, alpha = self.attention(encoder_out, output[:, -1, :])
        # weighted => [batch_size, encoder_dim]
        # alpha => [batch_size, num_pixels]

        combined = torch.cat([embedded.squeeze(1), weighted, output[:, -1, :]], dim=1)
        # combined => [batch_size, emb_dim + encoder_dim + hid_dim]

        logits = self.out(self.dropout(combined))
        # logits => [batch_size, vocab_size]

        return logits, hidden, cell, alpha



class Img2LaTeX(nn.Module):
    def __init__(self, encoder, decoder, encoder_dim, decoder_dim, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

    def init_hidden_state(self, encoder_out):
        # encoder_out => [batch_size, num_pixels, encoder_dim]
        mean_enc = torch.mean(encoder_out, dim=1)
        h = self.init_h(mean_enc)
        c = self.init_c(mean_enc)
        # h, c=> [batch_size, decoder_dim]

        return h, c
    
    def forward(self, images, formulas, lengths):
        # images => [batch_size, 3, 32, 128]
        # formulas => [batch_size, max_seq_len]
        # lengths => [batch_size]

        encoded_images = self.encoder(images)
        # encoded_images => [batch_size, 2, 8, 512]

        batch_size = encoded_images.size(0)
        encoder_dim = encoded_images.size(-1)
        encoder_out = encoded_images.view(batch_size, -1, encoder_dim)
        # encoder_
        hidden, cell = self.init_hidden_state(encoder_out)

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        # hidden, cell => [1, batch_size, decoder_dim]

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = [length - 1 for length in lengths]

        trg_len = formulas.size(1)
        output_dim = self.decoder.vocab_size
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        dec_inp = formulas[:, 0]

        for t in range(max(decode_lengths)):
            # calculate batch_size at each time step, so that only that part
            # will be used to train the model. (similar to pack padded sequence)
            batch_size_t = sum([l > t for l in decode_lengths])

            output, hidden, cell, alpha = self.decoder(
                dec_inp[:batch_size_t].unsqueeze(1),
                hidden[:, :batch_size_t, :],
                cell[:, :batch_size_t, :],
                encoder_out[:batch_size_t]
            )

            # save the output
            outputs[:batch_size_t, t, :] = output

            dec_inp = formulas[:batch_size_t, t]
        
        return outputs
