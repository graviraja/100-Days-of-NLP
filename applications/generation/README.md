<h1 align="center" style="font-size:80px">
    Generation based Applications in NLP
</h1>

There are many generation based problems in NLP like language modelling, machine translation, text summarization, sequence to sequence problems like NER, POS tagging, Image Captioning, and many others. Here I will be exploring a few of the applications.

**Note: Please raise an issue for any suggestions, corrections, and feedback.**

# Language Modeling

What is a Language Model in NLP? Language modeling is central to many important natural language processing tasks. A language model learns to predict the probability of a sequence of words.

## Generating Names.ipynb

Given a starting character, generate a name starting with that character. 

We’ll train LSTM character-level language models. That is, we’ll give the LSTM a huge chunk of names and ask it to model the probability distribution of the next character in the sequence given a sequence of previous characters. This will then allow us to generate new name one character at a time.

![name_gen](../../assets/images/applications/generation/name_gen.png)

As a working example, suppose we only had a vocabulary of all alphabets in English, and wanted to train an RNN on the training sequence "Jennie". This training sequence is in fact a source of 5 separate training examples: 
1. The probability of `e` should be likely given the context of `J`, 
2. `n` should be likely in the context of `Je`, 
3. `n` should also be likely given the context of `Jen`,
4. `i` should also be likely given the context of `Jenn`, 
and finally 
5. `e` should be likely given the context of `Jenni`.

#### Resources

- [Unreasonable effectiveness of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [US Baby Names Dataset](https://www.kaggle.com/kaggle/us-baby-names?select=NationalNames.csv)
- [Language Modeling - ChunML](https://github.com/ChunML/NLP/blob/master/text_generation/)

# Machine Translation

Machine Translation (MT) is the task of automatically converting one natural language into another, preserving the meaning of the input text, and producing fluent text in the output language. Ideally, a source language sequence is translated into target language sequence. 

## Basic Machine Translation.ipynb

The most common sequence-to-sequence (seq2seq) models are encoder-decoder models, which commonly use a recurrent neural network (RNN) to encode the source (input) sentence into a single vector. In this notebook, we'll refer to this single vector as a context vector. We can think of the context vector as being an abstract representation of the entire input sentence. This vector is then decoded by a second RNN which learns to output the target (output) sentence by generating it one word at a time.

![name_gen](../../assets/images/applications/generation/basic_translation.png)

#### Resources

- [Unreasonable effectiveness of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Ben Trevett Seq2Seq](https://github.com/bentrevett/pytorch-seq2seq)
- [Multi30K dataset](https://pytorch.org/text/datasets.html#multi30k)
- [Sequence to Sequence Learning with Neural Networks paper](https://arxiv.org/abs/1409.3215)

## Improved Machine Translation.ipynb

After trying the basic machine translation which has text perplexity `36.68`, following techniques have been experimented and a test perplexity `7.041`.

- GRU is used instead of LSTM
- Single layer
- Context vector is sent to decoder rnn along with decoder input embedding
- Context vector is sent to classifier along with the decoder hidden state

![improved_mt](../../assets/images/applications/generation/improved_mt.png)

#### Resources

- [Unreasonable effectiveness of RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Ben Trevett Seq2Seq](https://github.com/bentrevett/pytorch-seq2seq)
- [Multi30K dataset](https://pytorch.org/text/datasets.html#multi30k)
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation paper](https://arxiv.org/pdf/1406.1078.pdf)

## Machine Translation with Bahdanau Attention.ipynb

The attention mechanism was born to help memorize long source sentences in neural machine translation (NMT). Rather than building a single context vector out of the encoder's last hidden state, attention is used to focus more on the relevant parts of the input while decoding a sentence. The context vector will be created by taking encoder outputs and the `previous hidden state` of the decoder rnn.

![mt_bahdanau](../../assets/images/applications/generation/mt_bahdanau.png)

#### Resources

- [Ben Trevett Seq2Seq](https://github.com/bentrevett/pytorch-seq2seq)
- [Bahdanau Attention](https://github.com/graviraja/100-Days-of-NLP/blob/master/architectures/bahdanau_attention.py)
