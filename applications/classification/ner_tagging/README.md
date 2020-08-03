# NER Tagging

Named-Entity-Recognition (NER) tagging, is a task of labelling each word in a sentence with its appropriate entity.

## NER tagging with BiLSTM.ipynb

This code covers the basic workflow. We'll see how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop and train, test the model.

The model used is a Bi-directional LSTM network

![ner](../../../assets/images/applications/classification/ner_lstm.png)

The NER dataset is taken from [kaggle](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus) contains the following entities:

- `geo`: Geographical Entity
- `org`: Organization
- `per`: Person
- `gpe`: Geopolitical Entity
- `tim`: Time indicator
- `art`: Artifact
- `eve`: Event
- `nat`: Natural Phenomenon
- `o`: Other

A test accuracy of `96.7%` is achieved without the use of pre-trained embeddings. With that it might improve more.

*Note: Accuracy might not be a good metric as most of the data contains `O` tag. Different metrics like `precision`, `recall`, `f1` would give more insights.*

#### Resources

- [LSTM](https://github.com/graviraja/100-Days-of-NLP/blob/master/architectures/RNN.ipynb)
- [NER dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus)
