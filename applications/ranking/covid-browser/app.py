import os
from shutil import rmtree
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import scipy
import time
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import models, SentenceTransformer

st.write('# Covid-19 Browser')

CORPUS_PATH = 'corpus.pkl'
# Download the model
MODELS_DIR = 'models'
MODELS = {
    'scibert-nli': 'gsarti/scibert-nli',
    'biobert-nli': 'gsarti/biobert-nli',
    'covidbert-nli': 'gsarti/covidbert-nli',
    'clinicalcovidbert-nli':'manueltonneau/clinicalcovid-bert-nli'
}

model_name = st.selectbox(
    "Select the model",
    ('scibert-nli', 'biobert-nli', 'covidbert-nli', 'clinicalcovidbert-nli'),
    index=2
)

'#### Selected model:', model_name
EMBEDDINGS_PATH = f'{model_name}-embeddings.pkl'

path = os.path.join(MODELS_DIR, model_name)
if not os.path.exists(path):
    os.makedirs(path)

tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
model = AutoModel.from_pretrained(MODELS[model_name])
model.save_pretrained(path)
tokenizer.save_pretrained(path)

word_embedding_model = models.BERT(
        path,
        max_seq_length=512,
        do_lower_case=True)

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                        pooling_mode_mean_tokens=True,
                        pooling_mode_cls_token=False,
                        pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
rmtree(path)
model.save(path)
print(f'Model {model_name} available in {path}')

# Displaying data

@st.cache
def load_metadata():
    df = pd.read_csv('./metadata.csv')
    return df

df = load_metadata()

corpus = [str(c).strip().lower() for c in df['abstract'].values[:10000]]
titles = [str(t).strip() for t in df['title'].values[:10000]]

# Convert the abstracts to embeddings
if not os.path.exists(EMBEDDINGS_PATH):
    print("Computing and caching model embeddings for future use...")
    print(f"Computing embeddings for {len(corpus)} abstracts...")
    embeddings = model.encode(corpus, show_progress_bar=True)
    with open(EMBEDDINGS_PATH, 'wb') as file:
        pickle.dump(embeddings, file)
else:
    print("Loading model embeddings from", EMBEDDINGS_PATH, '...')
    with open(EMBEDDINGS_PATH, 'rb') as file:
        embeddings = pickle.load(file)

# Query Input form
def ask_question(query, model, corpus, corpus_embed, top_k=5):
    queries = [query]
    query_embeds = model.encode(queries, show_progress_bar=False)
    results = {"title": [], "abstract": [], "score": []}
    for query, query_embed in zip(queries, query_embeds):
        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        for count, (idx, distance) in enumerate(distances[0:top_k]):
            results['title'].append(titles[idx])
            results['abstract'].append(corpus[idx])
            results['score'].append(round(1 - distance, 4))
    return results

# Convert the query into embeddings
st.markdown('### Ask a query')

query = st.text_input('', '')

if len(query) > 0:
    start_time = time.time()
    results = ask_question(query, model, corpus, embeddings)
    st.table(results)
    end_time = time.time()
    print(f"query took {end_time - start_time} secs")
