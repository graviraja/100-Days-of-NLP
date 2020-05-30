import time
import math
import string
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# a b c => 3 4 5
# d e f g => 6 7 8 9

characters = 'abcdefghijklmnopqrstuvwxyz'

src_word2id = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
trg_word2id = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
for index, char in enumerate(characters):
    src_word2id[char] = index + 3
    trg_word2id[str(index + 3)] = index + 3

src_id2word = {id: char for char, id in src_word2id.items()}
trg_id2word = {id: char for char, id in trg_word2id.items()}
BATCH_SIZE = 64


def generate_sample(length):
    src_sample = []
    trg_sample = []

    for i in range(length):
        id = random.randint(3, len(src_id2word) - 1)
        src_sample.append(src_id2word[id])
        trg_sample.append(trg_id2word[id])
    return " ".join(src_sample).strip(), " ".join(trg_sample).strip()

def generate_data(num_samples):
    src = []
    trg = []

    for i in range(num_samples):
        length = random.randint(3, 10)
        src_sample, trg_sample = generate_sample(length)
        src.append(src_sample)
        trg.append(trg_sample)
    
    assert len(src) == len(trg)
    return src, trg

src, trg = generate_data(10000)
train_src, test_src, train_trg, test_trg = train_test_split(src, trg, test_size=0.1, random_state=42)
train_src, valid_src, train_trg, valid_trg = train_test_split(train_src, train_trg, test_size=0.2, random_state=42) # 0.9 * 0.2 = 0.18

print(f"Number of training examples: {len(train_src)}")
print(f"Number of validation examples: {len(valid_src)}")
print(f"Number of testing examples: {len(test_src)}")


class ToyDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

        assert len(src) == len(trg)
        self.length = len(src)
    
    def __getitem__(self, index):
        src_seq, trg_seq = self.preprocess(self.src[index], self.trg[index])
        return src_seq, trg_seq

    def __len__(self):
        return self.length
    
    def preprocess(self, src_sent, trg_sent):
        src_seq = [src_word2id['<sos>']] + [src_word2id[word] for word in src_sent.split()] + [src_word2id['<eos>']]
        trg_seq = [trg_word2id['<sos>']] + [trg_word2id[word] for word in trg_sent.split()] + [trg_word2id['<eos>']]

        return torch.Tensor(src_seq), torch.Tensor(trg_seq)


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs, trg_seqs = zip(*data)

    src_seqs, src_lengths = merge(src_seqs)
    trg_seqs, trg_lengths = merge(trg_seqs)

    return src_seqs, src_lengths, trg_seqs, trg_lengths


def get_loader(src_data, trg_data, train=True, batch_size=BATCH_SIZE):
    dataset = ToyDataset(src_data, trg_data)

    if train:
        shuffle = True
    else:
        shuffle = False
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return dataloader

train_loader = get_loader(train_src, train_trg, True, BATCH_SIZE)
valid_loader = get_loader(valid_src, valid_trg, False, BATCH_SIZE)
test_loader = get_loader(test_src, test_trg, False, BATCH_SIZE)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout, pad_token):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_token)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        # src => [seq_len, batch_size]

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # embedded => [seq_len, batch_size, emb_dim]

        output, hidden = self.rnn(embedded)
        # output => [seq_len, batch_size, hid_dim]
        # hidden => [1, batch_size, hid_dim]

        return output, hidden


class LuongAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.w = nn.Linear(hidden_size, hidden_size, bias=False)

        elif self.method == "concat":
            self.w = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        # hidden => [1, batch_size, hidden_size]
        # encoder_outputs => [src_len, batch_size, hidden_size]

        if self.method == "dot":
            attention_energies = self.dot(hidden, encoder_outputs)
        elif self.method == "general":
            attention_energies = self.general(hidden, encoder_outputs)
        elif self.method == "concat":
            attention_energies = self.concat(hidden, encoder_outputs)

        return F.softmax(attention_energies, dim=0)

    def dot(self, decoder_hidden, encoder_outputs):
        return torch.sum(decoder_hidden * encoder_outputs, dim=2)
    
    def general(self, decoder_hidden, encoder_outputs):
        energy = self.w(encoder_outputs)
        return torch.sum(decoder_hidden * energy, dim=2)
    
    def concat(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        decoder_hidden = decoder_hidden.expand(src_len, -1, -1)
        combined = torch.cat((decoder_hidden, encoder_outputs), dim=-1)
        energy = self.w(combined).tanh()
        return torch.sum(self.v * energy, dim=2)


class DecoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, attn, dropout, pad_token):
        super().__init__()

        self.output_dim = input_dim
        self.attn = attn
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_token)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.concat = nn.Linear(hid_dim * 2, hid_dim)
        self.out = nn.Linear(hid_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, hidden, enc_out):
        inputs = trg.unsqueeze(0)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        decoder_out, hidden = self.rnn(embedded, hidden)
        attn_weights = self.attn(hidden, enc_out)
        attn_weights = attn_weights.transpose(1, 0)
        enc_outs = enc_out.transpose(1, 0)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outs)
        combined = torch.cat((decoder_out, context.transpose(1, 0)), dim=2)
        middle = self.concat(combined)
        logits = self.out(middle.tanh().squeeze(0))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_force_ratio=0.5):
        encoder_output, hidden = self.encoder(src)
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, output_dim)
        dec_inp = trg[0, :]
        for i in range(1, trg_len):
            output, hidden = self.decoder(dec_inp, hidden, encoder_output)
            outputs[i] = output
            teacher_force = random.random() < teacher_force_ratio
            top1 = output.argmax(1)
            dec_inp = trg[i] if teacher_force else top1
        return outputs

INPUT_DIM = len(src_word2id)
OUTPUT_DIM = len(trg_word2id)
EMBEDDING_DIM = 10
HIDDEN_DIM = 20
DROPOUT = 0.5
PAD_TOKEN = trg_word2id['<pad>']
LUONG_METHOD = 'concat'

enc = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT, PAD_TOKEN)
attn = LuongAttention(LUONG_METHOD, HIDDEN_DIM)
dec = DecoderRNN(OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, attn, DROPOUT, PAD_TOKEN)
model = Seq2Seq(enc, dec)

def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model)} trainable parameters')

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)


def train(model, iterator, criterion, optimizer, clip):
    epoch_loss = 0
    model.train()

    for batch in iterator:
        src, src_lengths, trg, trg_lengths = batch
        src = src.transpose(1, 0)
        trg = trg.transpose(1, 0)
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            src, src_lengths, trg, trg_lengths = batch
            src = src.transpose(1, 0)
            trg = trg.transpose(1, 0)
            output = model(src, trg)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


def inference(src_sentence, model, max_len=10):
    model.eval()

    tokens = [src_word2id['<sos>']] + [src_word2id[word] for word in src_sentence.split()] + [src_word2id['<eos>']]
    src_tensor = torch.tensor(tokens).unsqueeze(1)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    trg_ids = [trg_word2id['<sos>']]
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_ids[-1]])
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_ids.append(pred_token)
        
        if pred_token == trg_word2id['<eos>']:
            break
    trg_seq = [trg_id2word[id] for id in trg_ids[1:]]
    trg_ids = [src_word2id[word] for word in src_sentence.split()]
    trg_org = [trg_id2word[id] for id in  trg_ids]
    predicted = " ".join(trg_seq).strip()
    truth = " ".join(trg_org).strip()
    print(f"Src sent: {src_sentence}, Predicted: {predicted}, Ground Truth: {truth}")


N_EPOCHS = 1000
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, criterion, optimizer, CLIP)
    valid_loss = evaluate(model, valid_loader, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    
    print(f"Epoch {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} |")
    print(f"\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f} |")
    sentence = "a g h i k"
    inference(sentence, model, len(sentence.split()))

model.load_state_dict(torch.load('model.pt'))
test_loss = evaluate(model, test_loader, criterion)
print(f"\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")
