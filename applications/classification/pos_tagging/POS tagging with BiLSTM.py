import time
import random
import spacy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data, datasets
import torchtext.vocab as vocab

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


TEXT = data.Field(lower = True)
UD_TAGS = data.Field(unk_token = None)
PTB_TAGS = data.Field(unk_token = None)

fields = (("text", TEXT), ("udtags", UD_TAGS), ("ptbtags", PTB_TAGS))

train_data, valid_data, test_data = datasets.UDPOS.splits(fields)

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

MIN_FREQ = 2

TEXT.build_vocab(
    train_data, 
    min_freq = MIN_FREQ)

# Uncomment the following for using pre-trained glove embeddings
# glove_file = 'glove.6B.50d.txt'
# glove_embeddings = vocab.Vectors(name=glove_file)
# TEXT.build_vocab(
#     train_data, 
#     min_freq=MIN_FREQ,
#     vectors=glove_embeddings,
#     unk_init=torch.Tensor.normal_)

UD_TAGS.build_vocab(train_data)
PTB_TAGS.build_vocab(train_data)


print(f"Tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Tokens in UD_TAG vocabulary: {len(UD_TAGS.vocab)}")
print(f"Tokens in PTB_TAG vocabulary: {len(PTB_TAGS.vocab)}")

BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE,
    device = device)


class POSTagger(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inp):
        # inp => [seq_len, batch_size]

        embedded = self.dropout(self.embedding(inp))
        # embedded => [seq_len, batch_size, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        outputs = self.dropout(outputs)
        # outputs => [seq_len, batch_size, hidden_dim * 2]

        logits = self.fc(outputs)
        # logits => [seq_len, batch_size, output_dim]

        return logits


INPUT_DIM = len(TEXT.vocab.itos)
EMB_DIM = 50
HIDDEN_DIM = 120
OUTPUT_DIM = len(UD_TAGS.vocab.itos)
N_LAYERS = 2
DROPOUT = 0.3
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = POSTagger(INPUT_DIM, EMB_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT, PAD_IDX)
model = model.to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# Uncomment these if pretrained vectors are used
# pretrained_embeddings = TEXT.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)
# model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD_IDX).to(device)


def categorical_accuracy(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])


def train(model, iterator, criterion, optimizer, tag_pad_idx, clip):
    model.train()

    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        text = batch.text
        tags = batch.udtags
        # text => [seq_len, batch_size]
        # tags => [seq_len, batch_size]

        optimizer.zero_grad()

        logits = model(text)
        # logits => [seq_len, batch_size, output_dim]

        logits = logits.view(-1, logits.shape[-1])
        # logits => [seq_len * batch_size, output_dim]

        tags = tags.view(-1)
        # tags => [seq_len * batch_size]

        loss = criterion(logits, tags)
        acc = categorical_accuracy(logits, tags, tag_pad_idx)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)



def evaluate(model, iterator, criterion, trg_pad_idx):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in iterator:
            text = batch.text
            tags = batch.udtags
            # text => [seq_len, batch_size]
            # tags => [seq_len, batch_size]

            optimizer.zero_grad()

            logits = model(text)
            # logits => [seq_len, batch_size, output_dim]

            logits = logits.view(-1, logits.shape[-1])
            # logits => [seq_len * batch_size, output_dim]

            tags = tags.view(-1)
            # tags => [seq_len * batch_size]

            loss = criterion(logits, tags)
            acc = categorical_accuracy(logits, tags, trg_pad_idx)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, criterion, optimizer, TAG_PAD_IDX, CLIP)
    valid_loss, val_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f} | Val. Loss: {valid_loss:.3f} | Val Acc: {val_acc * 100:.2f}')


model.load_state_dict(torch.load('model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}')


def inference(model, device, sentence, text_field, tags_field):
    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]
    
    if text_field.lower:
        tokens = [token.lower() for token in tokens]
    
    tokens_seq = [text_field.vocab.stoi[token] for token in tokens]
    unk_idx = text_field.vocab.stoi[text_field.pad_token]

    unks = [t for t, n in zip(tokens, tokens_seq) if n == unk_idx]

    tokens_tensor = torch.LongTensor(tokens_seq)
    # tokens_tensor => [seq_len]

    tokens_tensor = tokens_tensor.unsqueeze(-1).to(device)
    # tokens_tensor => [seq_len, 1 (batch_size)]

    logits = model(tokens_tensor)
    # logits => [seq_len, 1, output_dim]

    predictions = logits.argmax(-1)

    predicted_tags = [tags_field.vocab.itos[i] for i in predictions]

    return tokens, predicted_tags, unks


example_index = 1

sentence = vars(train_data.examples[example_index])['text']
actual_tags = vars(train_data.examples[example_index])['udtags']

print(sentence)

tokens, pred_tags, unks = inference(
    model, 
    device, 
    sentence, 
    TEXT, 
    UD_TAGS)


print("Pred. Tag\tActual Tag\tCorrect?\tToken\n")

for token, pred_tag, actual_tag in zip(tokens, pred_tags, actual_tags):
    correct = '✔' if pred_tag == actual_tag else '✘'
    print(f"{pred_tag}\t\t{actual_tag}\t\t{correct}\t\t{token}")

sentence = 'I love this movie'

tokens, tags, unks = inference(
    model, 
    device, 
    sentence, 
    TEXT, 
    UD_TAGS)


print("Pred. Tag\tToken\n")

for token, tag in zip(tokens, tags):
    print(f"{tag}\t\t{token}")