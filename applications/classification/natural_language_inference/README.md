# Natural Language Inference (NLI)

The goal of natural language inference (NLI), a widely-studied natural language processing task, is to determine if one given statement (a premise) semantically entails another given statement (a hypothesis).

## SNLI - Stanford Natural Language Inference Dataset

The SNLI corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE).

### NLI with BiLSTM.ipynb

This notebook covers the basic workflow. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop.

The model used is a Siamese BiLSTM network.

![nli](../../../assets/images/applications/classification/nli_bilstm.png)

This can be treated as base-line setup. A test accuracy of `76.84%` was achieved.

#### Resources

- [SNLI Dataset](https://nlp.stanford.edu/projects/snli/)

### NLI with Attention.ipynb

In the previous notebook, we have taken the final hidden states of Premise and Hypothesis as the representations from LSTM. Now instead of taking the final hidden states, attention will be computed across all the input tokens and a final weighted vector is taken as the representation of Premise and Hypothesis. 

![nli](../../../assets/images/applications/classification/nli_attention.png)

The test accuracy increased from `76.84%` to `79.51%`.

Also embeddings are initialized with Glove. In order to reduce the unneccessary computations `packing padded sequences` trick was also implemented. As a result, training time reduced.

#### Resources

- [Packing Padded Sequences](https://github.com/graviraja/100-Days-of-NLP/blob/master/architectures/pack_padded_sequences.py)
