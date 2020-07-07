import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import youtokentome

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiGRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, n_layers, pad_idx, dropout):
        super().__init__()
        
        self.n_layers = n_layers
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_length):
        # src => [seq_len, batch_size]
        # src_length => [batch_size]

        embedded = self.dropout(self.embedding(src))
        # embedded => [seq_len, batch_size, emb_dim]

        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, src_length)
        packed_outputs, hidden = self.rnn(packed_input)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs => [seq_len, batch_size, hidden_dim * 2]
        # hidden => [num_layers * num_dir, batch_size, hidden_dim]
        
        combined_outputs = outputs[:, :, :self.hid_dim] + outputs[:, :, self.hid_dim:]
        # combined_outputs => [seq_len, batch_size, hidden_dim]
        
        max_pooled = torch.max(combined_outputs, dim=0)[0]
        # max_pooled => [batch_size, hidden_dim]
        
        mean_pooled = torch.mean(combined_outputs, dim=0)
        # mean_pooled => [batch_size, hidden_dim]

        combined = torch.cat((mean_pooled, max_pooled), dim=-1)
        # combined => [batch_size, hidden_dim * 2]
        
        logits = torch.sigmoid(self.out(self.dropout(combined)))
        # logits => [batch_size, output_dim]

        return logits

bpe_model = youtokentome.BPE(model=os.path.join("model", "bpe.model"))
pad_index = bpe_model.subword_to_id('<PAD>')

PAD_IDX = pad_index
INPUT_DIM = bpe_model.vocab_size()
EMB_DIM = 100
HID_DIM = 256
N_LAYERS = 2
OUTPUT_DIM = 6
DROPOUT = 0.4

model = BiGRU(INPUT_DIM, EMB_DIM, HID_DIM, OUTPUT_DIM, N_LAYERS, PAD_IDX, DROPOUT).to(device)