# Toxic Comment Classification

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

## Setup

```code
pip install -r requirements.txt
```

## Running the application

Make sure the **`model.pt`** and **`bpe.model`** are present in the `model` folder.
```
streamlit run app.py
```

![utt_gen](../../../../assets/images/applications/classification/toxic_app.png)

## Further Improvements

The pre-trained model is a simple one which gives decent results. Following enhancements can be explored.

*   Pre-trained embeddings
*   Cross Validation
*   Ensemble of Networks
*   Data Augmentation
