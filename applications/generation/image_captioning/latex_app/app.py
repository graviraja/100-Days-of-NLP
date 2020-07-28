import os
from shutil import rmtree
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt

from src.model import Encoder, Decoder, Img2LaTeX

st.write('# Image to Latex Conversion')

device = torch.device('cpu')

PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

class Vocab(object):
    def __init__(self):
        self.sign2id = {
            "<s>": START_TOKEN,
            "</s>": END_TOKEN,
            "<pad>": PAD_TOKEN,
            "<unk>": UNK_TOKEN
        }

        self.id2sign = dict((id, sign) for sign, id in self.sign2id.items())
        self.length = 4
    
    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1
    
    def __len__(self):
        return self.length
    
    def __call__(self, sign):
        if not sign in self.sign2id:
            return self.sign2id['<unk>']
        return self.sign2id[sign]

def load_vocab():
    with open(os.path.join("model", 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    return vocab

@st.cache
def load_model():
    embed_dim = 256
    encoder_dim = 512
    decoder_dim = 512
    attention_dim = 512
    dropout = 0.3
    vocab = load_vocab()
    encoder = Encoder(encoder_dim).to(device)
    decoder = Decoder(len(vocab), embed_dim, decoder_dim, attention_dim, encoder_dim, dropout).to(device)
    model = Img2LaTeX(encoder, decoder, encoder_dim, decoder_dim, device).to(device)
    model.load_state_dict(torch.load(os.path.join("model", 'model.ckpt'), map_location=torch.device('cpu')))
    return model, vocab

def resize_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

def load_image(image_path, transform):
    image = Image.open(image_path)
    img = resize_image(image, [128, 32])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def greedy_generated_formula(image_path, model, max_len=38):
    # Prepare an image
    model.eval()
    transform = transforms.ToTensor()
    image = load_image(image_path, transform)
    image_tensor = image.to(device)
    
    with torch.no_grad():
        encoded_image = model.encoder(image_tensor)
        # encoded_image => [1, 2, 8, 512]
        
        batch_size = encoded_image.size(0)
        encoder_dim = encoded_image.size(-1)
        encoder_out = encoded_image.view(batch_size, -1, encoder_dim)
        # encoder_out => [1, 16, 512]
        
        num_pixels = encoder_out.size(1)

        hidden, cell = model.init_hidden_state(encoder_out)
        # hidden, cell => [1, decoder_dim]

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        # hidden, cell => [1, 1, decoder_dim]

        dec_inp = torch.LongTensor([vocab('<s>')]).to(device)
        sampled_ids = []

        for t in range(1, max_len):
            output, hidden, cell, _ = model.decoder(dec_inp.unsqueeze(1), hidden, cell, encoder_out)
            top1 = output.argmax(1)
            dec_inp = top1
            sampled_ids.append(top1.item())

    # Convert word_ids to words
    sampled_formula = []
    for sign_id in sampled_ids[1:]:
        sign = vocab.id2sign[sign_id]
        if sign == '</s>':
            break
        sampled_formula.append(sign)
    formula = ' '.join(sampled_formula)

    return formula


model, vocab = load_model()

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        greedy_formula = greedy_generated_formula(uploaded_file, model)
        st.write(f'### Input Image: \n')
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        st.write(f'### Generated formula')
        st.latex(greedy_formula)

    st.warning('Trained model is using only subset of the data. Using all the data and more training will improve the results!!')
