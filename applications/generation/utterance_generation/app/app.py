import os
import math
import youtokentome
import streamlit as st
import torch
import torch.nn.functional as F

from src.model import model

st.write('# Utterance Generation')

@st.cache
def load_model():
    model.load_state_dict(torch.load(os.path.join("model", 'model.pt'), map_location=torch.device('cpu')))

bpe_model = youtokentome.BPE(model=os.path.join("model", "bpe.model"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bpe_tokenizer(sentence):
    encoded_ids = bpe_model.encode(sentence.lower(), output_type=youtokentome.OutputType.ID, bos=True, eos=True)
    return encoded_ids


def generate_utterance_greedy(sentence, bpe_model, model, device, max_len=50):
    model.eval()

    if isinstance(sentence, str):
        tokens = bpe_tokenizer(sentence)
    else:
        tokens = [int(token) for token in sentence]

    src_indexes = tokens
 
    # convert to tensor format
    # since the inference is done on single sentence, batch size is 1
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    # src_tensor => [1, seq_len]

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    
    # the starting input to decoder is always <bos>
    trg_indexes = [bpe_model.subword_to_id('<BOS>')]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        # if the predicted token is <eos> means stop the decoding
        if pred_token == bpe_model.subword_to_id('<EOS>'):
            break
    
    # convert the predicted token ids to words
    trg_tokens = bpe_model.decode(trg_indexes, ignore_ids=[2,3])[0] # ignore <bos>, <eos>

    return tokens, trg_tokens, attention


def generate_utterance_beam(sentence, bpe_model, model, device, max_len=50, beam_size=10, length_norm_coefficient=0.6):
    with torch.no_grad():
        k = beam_size

        # minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # vocab size
        vocab_size = bpe_model.vocab_size()

        if isinstance(sentence, str):
            tokens = bpe_tokenizer(sentence)
        else:
            tokens = [int(token) for token in sentence]

        src_indexes = tokens
        
        # convert to tensor format
        # since the inference is done on single sentence, batch size is 1
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        # src_tensor => [1, seq_len]

        # encode
        enc_src = model.encoder(src_tensor)
        # enc_src => [1, src_len, d_model]

        # Our hypothesis to begin with is just <bos>
        hypotheses = torch.LongTensor([[bpe_model.subword_to_id('<BOS>')]]).to(device)  # (1, 1)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(device)  # (1)

        # Lists to store completed hypotheses and their scores
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # Start decoding
        step = 1

        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<sos>"
        while True:
            s = hypotheses.size(0)
            trg_mask = model.make_trg_mask(hypotheses)
            decoder_sequences, _ = model.decoder(hypotheses, enc_src.repeat(s, 1, 1), trg_mask)
            # decoder_sequences => [s, step_size, vocab_size]

            # Scores at this step
            scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # (s, vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                         dim=1)  # (k, step + 1)
            
            # Which of these new hypotheses are complete (reached <eos>)?
            complete = next_word_indices == bpe_model.subword_to_id('<EOS>')  # (k), bool

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # Stop if we have completed enough hypotheses
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # (s)

            # Stop if things have been going on for too long
            if step > 100:
                break
            step += 1
        
        # If there is not a single completed hypothesis, use partial hypotheses
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()
        
        # Decode the hypotheses
        all_hypotheses = list()
        for i, hypo in enumerate(completed_hypotheses):
            h = bpe_model.decode(hypo, ignore_ids=[2, 3])[0]    # ignore <bos>, <eos>
            all_hypotheses.append({"Utterance": h, "score": completed_hypotheses_scores[i]})
        
        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["Utterance"]

        return tokens, best_hypothesis, all_hypotheses


load_model()


if __name__ == "__main__":
    src = st.text_input("Enter the Sentence", "")

    if len(src) > 0:
        tokens, utterance, attention = generate_utterance_greedy(src, bpe_model, model, device)
        _, _, all_utterances = generate_utterance_beam(src, bpe_model, model, device)
        st.write(f'### Input: {src}\n')
        
        st.write(f'#### Greedy generated utterance')
        st.table([{'Utterance': utterance}])

        st.write("#### Beam generated utterances")
        st.table(all_utterances)
