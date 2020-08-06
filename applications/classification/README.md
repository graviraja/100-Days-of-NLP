<h1 align="center" style="font-size:80px">
    Classification based Applications in NLP
</h1>

There are many classification based problems in NLP like document classification, intent classification, question classification, sentiment analysis, emotion prediction and many others. Here I will be exploring a few of the applications.

**Note: Please raise an issue for any suggestions, corrections, and feedback.**

# Sentiment Analysis

Sentiment analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.

## Simple Sentiment Analysis.ipynb

This notebook covers the basic workflow. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop. The model used is a simple RNN network

## Improved Sentiment Analysis.ipynb

After trying the basic RNN which gives a test_accuracy less than 50%, following techniques have been experimented and a test_accuracy above 88% is achieved. The model used is a Multi Layer Bidirectional LSTM network

## Sentiment Analysis with Attention.ipynb

Attention helps in focusing on the relevant input when predicting the sentiment of the input. Bahdanau attention was used with taking the outputs of LSTM and concatenating the final forward & backward hidden state. Without using the pre-trained word embeddings, test accuracy of `88%` is achieved.

## Sentiment Analysis with BERT.ipynb

BERT obtains new state-of-the-art results on eleven natural language processing tasks. Transfer learning in NLP has triggered after the release of BERT model. In this notebook, we will explore how to use BERT for Sentiment Analysis.

# Document Classification

## Document Classification with Hierarchical Attention Network

A Hierarchical Attention Network (HAN) considers the hierarchical structure of documents (document - sentences - words) and includes an attention mechanism that is able to find the most important words and sentences in a document while taking the context into consideration.

# Quora Question Pairs classification

QQP stands for Quora Question Pairs. The objective of the task is for a given pair of questions; we need to find whether those questions are semantically similar to each other or not.

Quora released a dataset containing 400,000 pairs of potential question duplicate pairs. Each row contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair.

## QQP Classification with Siamese.ipynb

The algorithm needs to take the pair of questions as input and should output their similarity. A Siamese network is used. A `Siamese neural network` (sometimes called a twin neural network) is an artificial neural network that uses the `same weights` while working in tandem on two different input vectors to compute comparable output vectors.

## QQP Classification with BERT.ipynb

After trying the siamese model, BERT was explored to do the Quora duplicate question pairs detection. BERT takes the question 1 and question 2 as input separated by `[SEP]` token and the classification was done using the final representation of `[CLS]` token.

# POS Tagging

Part-of-Speech (PoS) tagging, is a task of labelling each word in a sentence with its appropriate part of speech.

## POS tagging with BiLSTM.py

This code covers the basic workflow. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop and run time (inference) tagging.

The model used is a Multi Layer Bi-directional LSTM network

## POS tagging with Transformer.py

After trying the RNN approach, POS tagging with Transformer based architecture is explored. Since the Transformer contains both Encoder and Decoder and for the sequence labeling task only `Encoder` will be sufficient. As the data is small having 6 layers of Encoder will overfit the data. So a 3-layer Transformer Encoder model was used.

## POS tagging with BERT.ipynb

After trying POS tagging with Transformer Encoder, POS Tagging with pre-trained BERT model is exploed. It achieved test accuracy of `91%`.

# Natural Language Inference (NLI)

The goal of natural language inference (NLI), a widely-studied natural language processing task, is to determine if one given statement (a premise) semantically entails another given statement (a hypothesis).

## NLI with BiLSTM.ipynb

This notebook covers the basic workflow. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop.

The model used is a Siamese BiLSTM network.

### NLI with Attention.ipynb

In the previous notebook, we have taken the final hidden states of Premise and Hypothesis as the representations from LSTM. Now instead of taking the final hidden states, attention will be computed across all the input tokens and a final weighted vector is taken as the representation of Premise and Hypothesis. 

The test accuracy increased from `76.84%` to `79.51%`.

### NLI with Transformer.ipynb

Transformer Encoder was used to encode the Premise and Hypothesis. Once the sentence is passed through the Encoder, summation of all the tokens is considered as the final representation (others variants can be explored). The model accuracy is less compared to RNN variants.

### NLI with BERT.ipynb

NLI with Bert base model was explored. BERT takes the Premise and Hypothesis as input separated by `[SEP]` token and the classification was done using the final representation of `[CLS]` token.

### NLI with Distillation.ipynb

**`Distillation`**: A technique you can use to compress a large model, called the `teacher`, into a smaller model, called the `student`. Following student, teacher models are used in order to perform distillation on NLI.

- Student Model: Logistic Regression
- Teacher Model: Bi-directional LSTM with Attention

# Toxic Comment Classification

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

### Toxic Comment Classification with GRU.ipynb

This notebook covers the basic workflow. We'll learn how to: load data, data analysis, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop.

The model used is a Bi-directional GRU network.

A test accuracy of `99.42%` was achieved. A test accuracy of `99.42%` was achieved. Also contains the implementation of ROC AUC metric.

### Improved Toxic Comment Classification.ipynb

With `Categorical Cross Entropy` as the loss, roc_auc score of `0.5` is achieved. By changing the loss to `Binary Cross Entropy` and also modifying the model a bit by adding pooling layers (max, mean), the roc_auc score improved to `0.9873`.

# Grammatically Correct Sentence

Can artificial neural networks have the ability to judge the grammatical acceptability of a sentence? In order to explore this task, the Corpus of Linguistic Acceptability (CoLA) dataset is used. CoLA is a set of sentences labeled as grammatically correct or incorrect. 

### CoLA with BERT.ipynb

BERT obtains new state-of-the-art results on eleven natural language processing tasks. Transfer learning in NLP has triggered after the release of BERT model. In this notebook, how to use BERT for classifying whether a sentence is grammatically correct or not using CoLA dataset is explored.


### CoLA with DistilBERT.ipynb

**`Distillation`**: A technique you can use to compress a large model, called the `teacher`, into a smaller model, called the `student`. Following student, teacher models are used in order to perform distillation on CoLA.

- Student Model: Distilbert base uncased
- Teacher Model: Bert base uncased

# NER Tagging

Named-Entity-Recognition (NER) tagging, is a task of labelling each word in a sentence with its appropriate entity.

## NER tagging with BiLSTM.ipynb

This code covers the basic workflow. We'll see how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop and train, test the model.

The model used is a Bi-directional LSTM network

## NER tagging with BiLSTM-CRF.ipynb

In the case of Sequence tagging (NER) the tag of a current word might depend on previous word's tag. (ex: New York).

Without a CRF, we would have simply used a single linear layer to transform the output of the Bidirectional LSTM into scores for each tag. These are known as `emission scores`, which are a representation of the likelihood of the word being a certain tag.

A CRF calculates not only the emission scores but also the `transition scores`, which are the likelihood of a word being a certain tag considering the previous word was a certain tag. Therefore the transition scores measure how likely it is to transition from one tag to another.

For decoding, `Viterbi` algorithm is used.

Since we're using CRFs, we're not so much predicting the right label at each word as we are predicting the right label sequence for a word sequence. Viterbi Decoding is a way to do exactly this – find the most optimal tag sequence from the scores computed by a Conditional Random Field.


## NER tagging with Char-BiLSTM-CRF.ipynb

Using sub-word information in our tagging task because it can be a powerful indicator of the tags, whether they're parts of speech or entities. For example, it may learn that adjectives commonly end with "-y" or "-ul", or that places often end with "-land" or "-burg".

Therefore, our sequence tagging model uses both

- `word-level` information in the form of word embeddings.
- `character-level` information up to and including each word in both directions.
