# Toxic Comment Classification

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

This is a kaggle [challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)


### Toxic Comment Classification with GRU.ipynb

This notebook covers the basic workflow. We'll learn how to: load data, data analysis, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop.

The model used is a Bi-directional GRU network.

![toxic](../../../assets/images/applications/classification/toxic_gru.png)

A test accuracy of `99.42%` was achieved. Also contains the implementation of ROC AUC metric.

### Improved Toxic Comment Classification.ipynb

With `Categorical Cross Entropy` as the loss, roc_auc score of `0.5` is achieved. By changing the loss to `Binary Cross Entropy` and also modifying the model a bit by adding pooling layers (max, mean), the roc_auc score improved to `0.9873`.

![toxic](../../../assets/images/applications/classification/improved_toxic.png)

#### Further Improvements

The following modifications/improvements can be explored.

*   Pre-trained embeddings
*   Cross Validation
*   Ensemble of Networks
*   Data Augmentation