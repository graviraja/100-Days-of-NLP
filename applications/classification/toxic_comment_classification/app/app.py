import os
import re
import math
import youtokentome
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd

import numpy as np
from src.model import model

st.write('# Toxic Comment Classification')

def load_model():
    model.load_state_dict(torch.load(os.path.join("model", 'model.pt'), map_location=torch.device('cpu')))

bpe_model = youtokentome.BPE(model=os.path.join("model", "bpe.model"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bpe_tokenizer(sentence):
    encoded_ids = bpe_model.encode(sentence.lower(), output_type=youtokentome.OutputType.ID)
    return encoded_ids


def preprocess(text):
    # -- Converting to lower case
    text = text.lower()
    
    # replacing english abbreviations with full forms
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    # text = re.sub('\W', ' ', text)
    # text = re.sub('\s+', ' ', text)

    text = " ".join(text.split()).strip()
    return text

def predict(sentence, bpe_model, model):
    model.eval()

    if isinstance(sentence, str):
        sentence = preprocess(sentence)
        tokens = bpe_tokenizer(sentence)
    else:
        tokens = [int(token) for token in sentence]

    src_indexes = tokens
 
    # convert to tensor format
    # since the inference is done on single sentence, batch size is 1
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    # src_tensor => [seq_len, 1]

    src_length = torch.LongTensor([len(src_indexes)])
    # src_length => [1]

    with torch.no_grad():
        predictions = model(src_tensor, src_length)

    return predictions


if __name__ == "__main__":
    src = st.text_input("Enter the Sentence", "")
    load_model()
    if len(src) > 0:
        predictions = predict(src, bpe_model, model)
        predictions = predictions.detach().cpu().numpy()
        st.write(f'### Input:')
        st.write(src)

        chart_data = pd.DataFrame(
            {
                "category": ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
                "score": predictions[0]
            })

        st.write(f'### Predicted Labels:')
        st.write("\n")
        st.table(chart_data)
