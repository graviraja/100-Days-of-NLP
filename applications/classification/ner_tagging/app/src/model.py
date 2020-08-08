import torch
import torch.nn as nn


class CharBiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, char_emb_dim, char_hid_dim, char_vocab_size, tag_vocab_size, sent_pad_token, tag_start_token, dropout=0.3):
        super().__init__()
        self.hid_dim = hid_dim
        self.sent_pad_token = sent_pad_token
        self.tag_start_token = tag_start_token
        self.tag_vocab_size = tag_vocab_size

        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_emb_dim, char_hid_dim, bidirectional=True, batch_first=True)

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim + char_hid_dim,
            hid_dim,
            bidirectional=True,
            batch_first=True
        )
        self.emission = nn.Linear(hid_dim * 2, tag_vocab_size)
        self.transition = nn.Parameter(torch.rand(tag_vocab_size, tag_vocab_size))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sentences, lengths, words, tags):
        # sentences => [batch_size, seq_len]
        # lengths => [batch_size]
        # words => [batch_size, seq_len, word_len]
        # tags => [batch_size, seq_len]

        char_final_hidden = []
        for word in words:
            # word => [seq_len, word_len]
            char_embed = self.char_embedding(word)
            char_embed = self.dropout(char_embed)
            # char_embed => [seq_len, word_len, char_emb_dim]

            _, (char_hidden, _) = self.char_lstm(char_embed)
            # char_hidden => [2, seq_len, char_hid_dim]

            # add the final forward and backward hidden states
            char_combined = char_hidden[-1, :, :] + char_hidden[-2, :, :]
            # char_combined => [seq_len, char_hid_dim]

            char_final_hidden.append(char_combined)
        
        char_encoding = torch.stack(char_final_hidden)
        # char_encoding => [batch_size, seq_len, char_hid_dim]

        mask = (sentences != self.sent_pad_token)
        # mask => [batch_size, seq_len]

        embed = self.embedding(sentences)
        embed = self.dropout(embed)
        # embed => [batch_size, seq_len, emb_dim]

        embed_with_char = torch.cat((embed, char_encoding), dim=-1)
        # embed_with_char => [batch_size, seq_len, emb_dim + char_hid_dim]

        packed_input = nn.utils.rnn.pack_padded_sequence(embed_with_char, lengths, batch_first=True)
        packed_output, _ = self.lstm(packed_input)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # outputs => [batch_size, seq_len, hid_dim * 2]

        combined = torch.cat((outputs[:, :, :self.hid_dim], outputs[:, :, self.hid_dim:]), dim=-1)
        combined = self.dropout(combined)
        # combined => [batch_size, seq_len, hid_dim * 2]

        emission_scores = self.emission(combined)
        # emission_scores => [batch_size, seq_len, tag_size]

        loss = self.vitebri_loss(tags, mask, emission_scores)
        # loss => [batch_size]

        return loss

    def vitebri_loss(self, tags, mask, emit_scores):
        # tags => [batch_size, seq_len]
        # mask => [batch_size, seq_len]
        # emit_scores => [batch_size, seq_len, tag_size]

        batch_size, sent_len = tags.shape

        # calculate the ground truth score
        score = torch.gather(emit_scores, 2, tags.unsqueeze(2)).squeeze(2)
        # emission scores of actual tags
        # score => [batch_size, seq_len]

        # add the transition scores to the emission scores
        # ignore the start token tag score
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]

        # consider only the scores of actual tokens not the padded
        gold_scores = (score * mask.type(torch.float)).sum(dim=1)
        # gold_scores => [batch_size]

        # calculate the scores of the partition (Z)
        # tensor to hold the accumulated sequence scores at each time step
        # at the inital time step score will be on dim=0
        scores_upto_t = emit_scores[:, 0].unsqueeze(1)
        # scores_upto_t => [batch_size, 1, tag_size]

        for i in range(1, sent_len):
            # get the current batch_size
            batch_t = mask[:, i].sum()

            # get the accumulated scores till now (only the current batch size)
            scores_unpad = scores_upto_t[:batch_t]
            # scores_unpad => [batch_t, 1, tag_size]

            # add the transition scores for this time step
            scores_with_trans = emit_scores[:batch_t, i].unsqueeze(1) + self.transition
            # scores_with_trans => [batch_t, tag_size, tag_size]

            # add to the accumulation
            sum_scores = scores_unpad.transpose(1, 2) + scores_with_trans
            # sum_scores => [batch_t, tag_size, tag_size]
            
            # apply the following to overcome the overflow problems
            # since the exp(some_big_number) will cause issues 
            # log(Σ exp(z_k)) = max(z) + log(Σ exp(z_k - max(z)))
            # log(Σ exp(z_k)) = log(Σ exp(z_k - c + c))
            #                 = log(Σ exp(z_k - c) * exp(c))
            #                 = log(Σ exp(z_k - c)) + log(exp(c))
            #                 = log(Σ exp(z_k - c)) + c
            # by taking c as max(z)
            # log(Σ exp(z_k)) = max(z) + log(Σ exp(z_k - max(z))) [log_sum_exp]
            # get the maximum score of the current time step
            max_t = sum_scores.max(dim=1)[0].unsqueeze(1)
            # max_t => [batch_t, 1, tag_size]

            sum_scores = sum_scores - max_t
            # sum_scores => [batch_t, tag_size, tag_size]

            scores_t = max_t + torch.logsumexp(sum_scores, dim=1).unsqueeze(1)
            # scores_t => [batch_t, 1, tag_size]

            # update the accumulation scores
            scores_upto_t = torch.cat((scores_t, scores_upto_t[batch_t:]), dim=0)
            # scores_upto_t => [batch_size, 1, tag_size]
        
        final_scores = scores_upto_t.squeeze(1)
        # final_scores => [batch_size, tag_size]

        max_final_scores = final_scores.max(dim=-1)[0]
        # max_final_scores => [batch_size]

        predicted_scores = max_final_scores + torch.logsumexp(final_scores - max_final_scores.unsqueeze(1), dim=1)
        # predicted_scores => [batch_size]

        vitebri_loss = predicted_scores - gold_scores
        # vitebri_loss => [batch_size]

        return vitebri_loss
    
    def predict(self, sentences, lengths, words):
        # sentences => [batch_size, seq_len]
        # lengths => [batch_size]
        # words => [batch_size, seq_len, word_len]

        batch_size = sentences.size(0)

        char_final_hidden = []
        for word in words:
            # word => [seq_len, word_len]
            char_embed = self.char_embedding(word)
            char_embed = self.dropout(char_embed)
            # char_embed => [seq_len, word_len, char_emb_dim]

            _, (char_hidden, _) = self.char_lstm(char_embed)
            # char_hidden => [2, seq_len, char_hid_dim]

            # add the final forward and backward hidden states
            char_combined = char_hidden[-1, :, :] + char_hidden[-2, :, :]
            # char_combined => [seq_len, char_hid_dim]

            char_final_hidden.append(char_combined)
        
        char_encoding = torch.stack(char_final_hidden)
        # char_encoding => [batch_size, seq_len, char_hid_dim]

        mask = (sentences != self.sent_pad_token)
        # mask => [batch_size, seq_len]

        embed = self.embedding(sentences)
        embed = self.dropout(embed)
        # embed => [batch_size, seq_len, emb_dim]

        embed_with_char = torch.cat((embed, char_encoding), dim=-1)
        # embed_with_char => [batch_size, seq_len, emb_dim + char_hid_dim]

        packed_inp = nn.utils.rnn.pack_padded_sequence(embed_with_char, lengths, batch_first=True)
        packed_output, _ = self.lstm(packed_inp)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # outputs => [batch_size, seq_len, hid_dim * 2]

        combined = torch.cat((outputs[:, :, :self.hid_dim], outputs[:, :, self.hid_dim:]), dim=-1)
        combined = self.dropout(combined)
        # combined => [batch_size, seq_len, hid_dim * 2]

        emission_scores = self.emission(combined)
        # emission_scores => [batch_size, seq_len, tag_size]

        # to store the tags predicted at each time step
        # since at the begining every tag is start tag create the list with start tags
        tags = [[[self.tag_start_token] for _ in range(self.tag_vocab_size)]] * batch_size
        # tags => [batch_size, tag_size, 1]

        scores_upto_t = emission_scores[:, 0].unsqueeze(1)
        # scores_upto_t => [batch_size, 1, tag_size]

        for i in range(1, max(lengths)):
            # get the current batch_size
            batch_t = mask[:, i].sum()

            # get the accumulated scores till now (only the current batch size)
            scores_unpad = scores_upto_t[:batch_t]
            # scores_unpad => [batch_t, 1, tag_size]

            # add the transition scores for this time step
            scores_with_trans = emission_scores[:batch_t, i].unsqueeze(1) + self.transition
            # scores_with_trans => [batch_t, tag_size, tag_size]

            # add to the accumulation
            sum_scores = scores_unpad.transpose(1, 2) + scores_with_trans
            # sum_scores => [batch_t, tag_size, tag_size]

            max_scores_t, max_ids_t = torch.max(sum_scores, dim=1)
            max_ids_t = max_ids_t.tolist()
            # max_scores_t => [batch_t, tag_size]
            # max_ids_t => [batch_t, tag_size]

            # add the current time step predicted tags 
            tags[:batch_t] = [[tags[b][k] + [j] for j, k in enumerate(max_ids_t[b])] for b in range(batch_t)]
            
            # update the accumulation scores
            scores_upto_t = torch.cat((max_scores_t.unsqueeze(1), scores_upto_t[batch_t:]), dim=0)
            # scores_upto_t => [batch_size, tag_size]

        scores = scores_upto_t.squeeze(1)
        # scores => [batch_size, tag_size]

        _, max_ids = torch.max(scores, dim=1)
        max_ids = max_ids.tolist()
        # max_ids => [batch_size]

        # tags => [batch_size, tag_size, seq_len]
        tags = [tags[b][k] for b, k in enumerate(max_ids)]
        # tags => [batch_size, seq_len]

        return tags
