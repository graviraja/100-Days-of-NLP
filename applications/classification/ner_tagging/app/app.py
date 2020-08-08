import os
import pickle
import streamlit as st
import torch

from src.model import CharBiLSTMCRF

st.write('# Named Entity Recognition')

MAX_WORD_LEN = 15

class Vocab:
    def __init__(self, word2id, id2word):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.__word2id = word2id
        self.__id2word = id2word

    def get_word2id(self):
        return self.__word2id

    def get_id2word(self):
        return self.__id2word

    def __getitem__(self, item):
        if self.UNK in self.__word2id:
            return self.__word2id.get(item, self.__word2id[self.UNK])
        return self.__word2id[item]

    def __len__(self):
        return len(self.__word2id)

    def id2word(self, idx):
        return self.__id2word[idx]


with open(os.path.join('models', 'chars_vocab.pkl'), 'rb') as f:
    chars_vocab = pickle.load(f)

with open(os.path.join('models', 'tags_vocab.pkl'), 'rb') as f:
    tags_vocab = pickle.load(f)

with open(os.path.join('models', 'words_vocab.pkl'), 'rb') as f:
    words_vocab = pickle.load(f)


@st.cache
def load_model():
    vocab_size = len(words_vocab)
    sent_pad_token = words_vocab[words_vocab.PAD]
    tag_start_token = tags_vocab[tags_vocab.START]
    emb_dim = 50
    hid_dim = 200
    char_emb_dim = 20
    char_hid_dim = 50
    char_vocab_size = len(chars_vocab)
    tag_vocab_size = len(tags_vocab)
    model = CharBiLSTMCRF(
        vocab_size,
        emb_dim,
        hid_dim,
        char_emb_dim,
        char_hid_dim,
        char_vocab_size,
        tag_vocab_size,
        sent_pad_token,
        tag_start_token
    )
    model.load_state_dict(torch.load(os.path.join("models", 'model.pt'), map_location=torch.device('cpu')))
    return model


def inference(sentence):
    if isinstance(sentence, str):
        tokens = [words_vocab[words_vocab.START]] + sentence.split() + [words_vocab[words_vocab.END]]
    else:
        tokens = sentence
    
    chars = [['<START']] + [['<START>'] + [ch for ch in word] + ['<END>'] for word in tokens[1:-1]] + [['<END>']]

    char_seq = []
    for word in chars:
        word_len = len(word)
        # truncate the word if it is greater than max_word_len
        if word_len > MAX_WORD_LEN:
            word = word[:MAX_WORD_LEN]
        # pad the word if it less
        else:
            pad_length = MAX_WORD_LEN - word_len
            word = word + [chars_vocab.PAD] * pad_length
        
        # convert the chars into numerical format
        char_ids = []
        for each_char in word: 
            char_ids.append(chars_vocab[each_char])
        char_seq.append(char_ids)

    # numericalize
    token_ids = [words_vocab[tok] for tok in tokens]
    
    # seq length
    sent_length = [len(token_ids)]

    # create tensors
    sent_tensor = torch.LongTensor(token_ids)
    sent_tensor = sent_tensor.unsqueeze(0)
    # sent_tensor => [1, seq_len]

    char_tensor = torch.LongTensor(char_seq)
    char_tensor = char_tensor.unsqueeze(0)
    # char_tensor => [1, seq_len, word_len]

    model.eval()
    with torch.no_grad():
        predictions = model.predict(sent_tensor, sent_length, char_tensor)
    
    predictions = predictions[0]
    predicted_tags = []
    for i in predictions:
        predicted_tags.append(tags_vocab.id2word(i))
    
    return tokens[1:-1], predicted_tags[1:-1]

model = load_model()

if __name__ == "__main__":
    src = st.text_input("Enter the Sentence", "")

    if len(src) > 0:
        tokens, predicted_tags = inference(src)
        st.write(f'### Input: \n {src} \n')

        st.write(f"#### Predicted tags: \n")
        result = {"tokens": tokens, "tags": predicted_tags}
        st.table(result)
