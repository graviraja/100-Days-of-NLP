import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_data(data_size, max_num, seq_len):
    x = np.array(
        [np.random.choice(range(max_num), size=seq_len, replace=False)
        for _ in range(data_size)])
    y = np.argsort(x)

    return torch.LongTensor(x), torch.LongTensor(y)

datasamples = 12000
max_num = 100
seq_len = 5
x, y = get_data(datasamples, max_num, seq_len)

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)

train_data = TensorDataset(train_x, train_y)
val_data = TensorDataset(val_x, val_y)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_size, num_layers, bidirectional):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
    
    def forward(self, inputs):
        # inputs => [batch_size, seq_len]

        embedded_inputs = self.embedding(inputs)
        # embedded_inputs => [batch_size, seq_len, emb_dim]

        outputs, hidden = self.rnn(embedded_inputs)
        # outputs => [batch_size, seq_len, hidden_dim * 2]
        # hidden => [num_layers * 2, batch_size, hidden_dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, encoder_outputs, hidden, mask=None):
        # encoder_outputs => [batch_size, seq_len, hidden_dim]
        # hidden => [batch_size, 1, hidden_dim]

        encoder_energy = self.w1(encoder_outputs)
        # encoder_energy => [batch_size, seq_len, hidden_dim]

        decoder_energy = self.w2(hidden.squeeze(1))
        # decoder_energy => [batch_size, hidden_dim]

        decoder_energy = decoder_energy.unsqueeze(1)
        # decoder_energy => [batch_size, 1, hidden_dim]

        combined = torch.tanh(encoder_energy + decoder_energy)
        # combined => [batch_size, seq_len, hidden_dim]

        energy = self.v(combined)
        # energy => [batch_size, seq_len, 1]

        energy = energy.squeeze(-1)
        # energy => [batch_size, seq_len]

        if mask is not None:
            energy = energy.masked_fill(mask, -1e10)

        attention = torch.softmax(energy, dim=-1)
        # attention => [batch_size, seq_len]

        return attention


class PointerNetwork(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_size, num_layers, bidirectional=True):
        super().__init__()

        self.embedding_dim = emb_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.encoder = Encoder(input_dim, emb_dim, hidden_size, num_layers, bidirectional)
        self.attn = Attention(hidden_size)
        self.decoder = nn.LSTM(emb_dim, hidden_size, num_layers=num_layers, batch_first=True)

    
    def forward(self, inputs):
        # inputs => [batch_size, seq_len]

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        encoder_outputs, encoder_states = self.encoder(inputs)
        # encoder_outputs => [batch_size, seq_len, hidden_dim * 2]
        # encoder_hidden, encoder_cell = encoder_states
        # encoder_hidden => [num_layers * num_dir, batch_size, hidden_dim]
        # encoder_cell => [num_layers * num_dir, batch_size, hidden_dim]

        encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        # encoder_outputs => [batch_size, seq_len, hidden_dim]

        encoder_hidden, encoder_cell = encoder_states
        encoder_hidden = encoder_hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = encoder_hidden[:, 0, :, :] + encoder_hidden[:, 1, :, :]
        # hidden => [num_layers, batch_size, hidden_dim]

        encoder_cell = encoder_cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = encoder_cell[:, 0, :, :] + encoder_cell[:, 1, :, :]
        # cell => [num_layers, batch_size, hidden_dim]

        decoder_input = torch.zeros(batch_size).long().unsqueeze(1)
        # decoder_input => [batch_size, 1]
        logits = torch.zeros(batch_size, seq_len, seq_len)

        for i in range(seq_len):
            dec_inp = self.encoder.embedding(decoder_input)
            # dec_inp => [batch_size, 1, emb_dim]
            output, (hidden, cell) = self.decoder(dec_inp, (hidden, cell))

            attention = self.attn(encoder_outputs, output)
            # attention => [batch_size, seq_len]
            logits[:, :, i] = attention
        return logits

emb_dim = 5
hidden_size = 15
num_layers = 2
model = PointerNetwork(max_num, emb_dim, hidden_size, num_layers)
clip = 1
lr = 2e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


def train(model, iterator, criterion, optimizer, clip):
    model.train()
    train_loss = 0

    for batch in iterator:
        src, target = batch
        # src => [batch_size, seq_len]
        # target => [batch_size, seq_len]

        logits = model(src)
        # logits => [batch_size, seq_len, seq_len]
        loss = criterion(logits, target)
        train_loss += loss.item()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return train_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    eval_loss = 0
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for batch in iterator:
            src, target = batch
            # src => [batch_size, seq_len]
            # target => [batch_size, seq_len]

            logits = model(src)
            # logits => [batch_size, seq_len, seq_len]

            loss = criterion(logits, target)
            eval_loss += loss.item()

            _, preds = torch.max(logits, dim=1)
            correct_count += preds.eq(target).sum().item()
            total_count += target.numel()

    eval_acc = float(correct_count) / float(total_count)
    return eval_loss / len(iterator), eval_acc


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


EPOCHS = 200
best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, criterion, optimizer, clip)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    end_time = time.time()

    elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}| Time: {elapsed_mins}m {elapsed_secs}s")

# model.load_state_dict(torch.load('model.pt'))

def inference(input_seq, model):
    input_seq = torch.LongTensor(input_seq)
    # input_seq => [1, seq_len]

    logits = model(input_seq)
    # logits => [1, seq_len, seq_len]

    logits = logits.squeeze()
    # logits => [seq_len, seq_len]

    _, predictions = torch.max(logits, dim=1)
    # predictions = [seq_len]

    return predictions

test =  [np.random.choice(range(max_num), size=seq_len, replace=False)]
test_trg = np.argsort(test)
predictions = inference(test, model)
print(f"Actual Seq: {test}")
print(f"Ground truth: {test_trg[0]}")
print(f"Predicted: {predictions}")