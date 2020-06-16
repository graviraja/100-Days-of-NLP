# POS Tagging

Part-of-Speech (PoS) tagging, is a task of labelling each word in a sentence with its appropriate part of speech.

## POS tagging with BiLSTM.py

This code covers the basic workflow. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop and run time (inference) tagging.

The model used is a Multi Layer Bi-directional LSTM network

![sentiment](../../../assets/images/applications/classification/pos_lstm.png)

A test accuracy of `88%` is achieved without the use of pre-trained embeddings. With that it might improve more.

#### Resources

- [LSTM](https://github.com/graviraja/100-Days-of-NLP/blob/master/architectures/RNN.ipynb)
- [POS tagging](https://github.com/bentrevett/pytorch-pos-tagging/)
- [UDPOS dataset](https://pytorch.org/text/datasets.html#udpos)
