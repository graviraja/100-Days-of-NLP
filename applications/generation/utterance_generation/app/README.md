# Utterance Generation

Utterance generation is an important problem in NLP, especially in question answering, information retrieval, information extraction, conversation systems, to name a few. It could also be used to create synthentic training data for many NLP problems.

## Setup

```code
pip install -r requirements.txt
```

## Running the application

Make sure the **`model.pt`** and **`bpe.model`** are present in the `model` folder.
```
streamlit run app.py
```

![utt_gen](../../../../assets/images/applications/generation/utt_gen_app.png)
