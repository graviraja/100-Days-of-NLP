
![emb](assets/images/embeddings/header.png)


 > There is nothing magic about magic. The magician merely understands something simple which doesn’t appear to be simple or natural to the untrained audience. Once you learn how to hold a card while making your hand look empty, you only need practice before you, too, can “do magic.” – Jeffrey Friedl in the book Mastering Regular Expressions

**Note: Please raise an issue for any suggestions, corrections, and feedback.**

Most of the code samples are done using Jupyter notebooks (using colab). So each code can be run independently.

The following topics have been explored:

<ul>
    <li><a href="#nlp-concepts">NLP Concepts</a></li>
    <li><a href="#classification-based-applications">Classification based Applications</a></li>
    <li><a href="#generation-based-applications">Generation based Applications</a></li>
    <li><a href="#clustering-based-applications">Clustering based Applications</a></li>
    <li><a href="#question-answering-based-applications">Question-Answering based Applications</a></li>
    <li><a href="#ranking-based-applications">Ranking based Applications</a></li>
    <li><a href="#recommendation-based-applications">Recommendation based Applications</a></li>
</ul>

*Note: The difficulty level has been assigned according to my understanding.*

# NLP Concepts


<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#tokenization">Tokenization</a></td>
    <td style="padding:15px;"><a href="#word-embeddings---word2vec">Word Embeddings - Word2Vec</a></td>
    <td style="padding:15px;"><a href="#word-embeddings---glove">Word Embeddings - GloVe</a></td>
    <td style="padding:15px;"><a href="#word-embeddings---elmo">Word Embeddings - ELMo</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#rnn-lstm-gru">RNN, LSTM, GRU</a></td>
    <td style="padding:15px;"><a href="#packing-padded-sequences">Packing Padded Sequences</a></td>
    <td style="padding:15px;"><a href="#attention-mechanism---luong">Attention Mechanism - Luong</a></td>
    <td style="padding:15px;"><a href="#attention-mechanism---bahdanau">Attention Mechanism - Bahdanau</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#pointer-network">Pointer Network</a></td>
    <td style="padding:15px;"><a href="#transformer">Transformer</a></td>
    <td style="padding:15px;"><a href="#gpt-2">GPT-2</a></td>
    <td style="padding:15px;"><a href="#bert">BERT</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#topic-modelling-using-lda">Topic Modeling - LDA</a></td>
    <td style="padding:15px;"><a href="#principal-component-analysispca">Principal Component Analysis (PCA)</a></td>
    <td style="padding:15px;"><a href="#naive-bayes-algorithm">Naive Bayes</a></td>
    <td style="padding:15px;"><a href="#data-augmentation-in-nlp">Data Augmentation</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#sentence-embeddings">Sentence Embeddings</a></td>
</tr>

</table>

## Tokenization

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>


The process of converting textual data to tokens, is one of the most important step in NLP. Tokenization using the following methods has been explored:
- [Spacy](https://spacy.io/usage/linguistic-features#tokenization)
- [Byte Pair Encoding (Sentencepiece)](https://github.com/google/sentencepiece)
- [Unigram Encoding (Sentencepiece)](https://github.com/google/sentencepiece)
- [Torchtext](https://pytorch.org/text/data_utils.html)
- [Tokenizers](https://github.com/huggingface/tokenizers)

## Word Embeddings - Word2Vec

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

A word embedding is a learned representation for text where words that have the same meaning have a similar representation. It is this approach to representing words and documents that may be considered one of the key breakthroughs of deep learning on challenging natural language processing problems.

![emb](assets/images/embeddings/embeddings.png)

Word2Vec is one of the most popular pretrained word embeddings developed by Google. Depending on the way the embeddings are learned, Word2Vec is classified into two approaches:

- Continuous Bag-of-Words (CBOW)
- Skip-gram model

![word2vec arch](assets/images/embeddings/word2vec.png)

## Word Embeddings - GloVe

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

GloVe is another commonly used method of obtaining pre-trained embeddings. GloVe aims to achieve two goals:

- Create word vectors that capture meaning in vector space
- Takes advantage of global count statistics instead of only local information

## Word Embeddings - ELMo

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

ELMo is a deep contextualized word representation that models:

- complex characteristics of word use (e.g., syntax and semantics)
- how these uses vary across linguistic contexts (i.e., to model polysemy).

These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus.

![elmo arch](assets/images/embeddings/elmo.png)

## RNN, LSTM, GRU

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Recurrent networks - RNN, LSTM, GRU have proven to be one of the most important unit in NLP applications because of their architecture. There are many problems where the sequence nature needs to be remembered like in order to predict an emotion in the scene, previous scenes needs to be remembered.

![rnn gif](./assets/images/architectures/rnn.gif)

## Packing Padded Sequences

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

When training RNN (LSTM or GRU or vanilla-RNN), it is difficult to batch the variable-length sequences. Ideally we will pad all the sequences to a fixed length and end up doing un-necessary computations. How can we overcome this? PyTorch provides the `pack_padded_sequences` functionality.

![pack img](./assets/images/architectures/pack_padded_seq.jpg)


## Attention Mechanism - Luong

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

The attention mechanism was born to help memorize long source sentences in neural machine translation (NMT). Rather than building a single context vector out of the encoder's last hidden state, attention is used to focus more on the relevant parts of the input while decoding a sentence. The context vector will be created by taking encoder outputs and the `current output` of the decoder rnn.

![pack img](./assets/images/architectures/luong_attention.png)

The attention score can be calculated in three ways. `dot`, `general` and `concat`.

![luong_fn](./assets/images/architectures/luong_fn.png)


## Attention Mechanism - Bahdanau

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

The major difference between Bahdanau & Luong attention is the way the context vector is created. The context vector will be created by taking encoder outputs and the `previous hidden state` of the decoder rnn. Where is in Luong attention the context vector will be created by taking encoder outputs and the `current hidden state` of the decoder rnn.

Once the context is calculated it is combined with decoder input embedding and fed as input to decoder rnn.

![pack img](./assets/images/architectures/bahdanau_attention.png)

The Bahdanau attention is also called as `additive` attention.

![bahdanau_fn](./assets/images/architectures/bahdanau_fn.jpg)


## Transformer

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. Such attention mechanisms are used in conjunction with a recurrent network.

The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

![transformer](./assets/images/architectures/transformer.png)


## GPT-2

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>
The GPT-2 paper states that: 

> Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText. Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText. Samples from the model reflect these improvements and contain coherent paragraphs of text. These findings suggest a promising path towards building language processing systems which learn to perform tasks from their naturally occurring demonstrations.

![gpt2](./assets/images/architectures/gpt2_usecase.png)

The GPT-2 utilizes a 12-layer Decoder Only Transformer architecture.

![gpt2](./assets/images/architectures/gpt2.png)


## BERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>
Language modeling is an effective task for using unlabeled data to pretrain neural networks in NLP. Traditional language models take the previous n tokens and predict the next one. In contrast, BERT trains a language model that takes both the previous and next tokens into account when predicting. BERT is also trained on a next sentence prediction task to better handle tasks that require reasoning about the relationship between two sentences (e.g. similar questions or not)

![bert](./assets/images/architectures/bert.png)

BERT uses the Transformer architecture for encoding sentences.

![bert](./assets/images/architectures/bert_arch.png)


## Pointer Network

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>
Pointer networks are sequence-to-sequence models where the output is discrete tokens corresponding to positions in an input sequence. The main differences between pointer networks and standard seq2seq models are:

- The output of pointer networks is discrete and correspond to positions in the input sequence

- The number of target classes in each step of the output depends on the length of the input, which is variable.

It differs from the previous attention attempts in that, instead of using attention to blend hidden units of an encoder to a context vector at each decoder step, it uses attention as a pointer to select a member of the input sequence as the output.

![pointer](./assets/images/architectures/pointer_network.png)


## Topic Modelling using LDA

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

One of the primary applications of natural language processing is to automatically extract what topics people are discussing from large volumes of text. Some examples of large text could be feeds from social media, customer reviews of hotels, movies, etc, user feedbacks, news stories, e-mails of customer complaints etc.

Knowing what people are talking about and understanding their problems and opinions is highly valuable to businesses, administrators, political campaigns. And it’s really hard to manually read through such large volumes and compile the topics.

Thus is required an automated algorithm that can read through the text documents and automatically output the topics discussed.

In this notebook, we will take a real example of the `20 Newsgroups` dataset and use LDA to extract the naturally discussed topics.

![lda](assets/images/architectures/lda.png)

LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.

Once you provide the algorithm with the number of topics, all it does it to rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.

## Principal Component Analysis(PCA)

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

PCA is fundamentally a dimensionality reduction technique that transforms the columns of a dataset into a new set features. It does this by finding a new set of directions (like X and Y axes) that explain the maximum variability in the data. This new system coordinate axes is called Principal Components (PCs).

![pca](assets/images/architectures/pca.png)

Practically PCA is used for two reasons:

- **`Dimensionality Reduction`**: The information distributed across a large number of columns is transformed into principal components (PC) such that the first few PCs can explain a sizeable chunk of the total information (variance). These PCs can be used as explanatory variables in Machine Learning models.

- **`Visualize Data`**: Visualising the separation of classes (or clusters) is hard for data with more than 3 dimensions (features). With the first two PCs itself, it’s usually possible to see a clear separation.


## Naive Bayes Algorithm

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.

![naive](assets/images/architectures/naive_bayes.png)

Using Bayes theorem, we can find the probability of A happening, given that B has occurred. Here, B is the evidence and A is the hypothesis. The assumption made here is that the predictors/features are independent. That is presence of one particular feature does not affect the other. Hence it is called naive.

**Types of Naive Bayes Classifier**:

`Multinomial Naive Bayes`:
This is mostly used when the variables are discrete (like words). The features/predictors used by the classifier are the frequency of the words present in the document.

`Gaussian Naive Bayes`:
When the predictors take up a continuous value and are not discrete, we assume that these values are sampled from a gaussian distribution.

`Bernoulli Naive Bayes`:
This is similar to the multinomial naive bayes but the predictors are boolean variables. The parameters that we use to predict the class variable take up only values yes or no, for example if a word occurs in the text or not.

Using 20newsgroup dataset, naive bayes algorithm is explored to do the classification.


## Data Augmentation in NLP

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>
In Computer Vision using image data augmentation is a standard practice. This is because trivial operations for images like rotating an image a few degrees or converting it into grayscale doesn’t change its semantics. Whereas in natural language processing (NLP) field, it is hard to augmenting text due to high complexity of language.

Data Augmentation using the following techniques is explored:

- Synonym-based Substitution
- Antonym-based Substitution
- Back Translation
- Text Surface Transformation
- Random Noise Injection
- Word Embedding based Substitution
- Contextual Word Embeddings (BERT family) based Substitution

![aug](assets/images/architectures/augmentation.png)


## Sentence Embeddings

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

A new architecture called SBERT was explored. The siamese network architecture enables that fixed-sized vectors for input sentences can be derived. Using a similarity measure like cosinesimilarity or Manhatten / Euclidean distance, semantically similar sentences can be found.

![sentence emb](assets/images/embeddings/sentence_emb.png)


# Classification based Applications

<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#sentiment-analysis---imdb">Sentiment Analysis - IMDB</a></td>
    <td style="padding:15px;"><a href="#sentiment-analysis---hinglish">Sentiment Classification - Hinglish</a></td>
    <td style="padding:15px;"><a href="#document-classification">Document Classification</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#duplicate-question-pair-classification---quora">Duplicate Question Pair Classification - Quora</a></td>
    <td style="padding:15px;"><a href="#pos-tagging">POS Tagging</a></td>
    <td style="padding:15px;"><a href="#natural-language-inference---snli">Natural Language Inference - SNLI</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#toxic-comment-classification">Toxic Comment Classification</a></td>
    <td style="padding:15px;"><a href="#grammatically-correct-sentence---cola">Grammatically Correct Sentence - CoLA</a></td>
    <td style="padding:15px;"><a href="#ner-tagging">NER Tagging</a></td>
</tr>

</table>

## Sentiment Analysis - IMDB

Sentiment analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.

Following varients have been explored:

- <a href="#simple-sentiment-classification-with-rnn">Simple Sentiment Analysis using RNN</a>
- <a href="#improved-sentiment-classification---imdb">Improved Sentiment Analysis</a>
- <a href="#sentiment-analysis-with-attention">Sentiment Analysis with Attention</a>
- <a href="#sentiment-analysis-with-bert">Sentiment Analysis with BERT</a>

### Simple Sentiment Classification with RNN

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

RNN is used for processing and identifying the sentiment.

![sentiment](assets/images/applications/sentiment/simple.gif)


### Improved Sentiment Classification

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

After trying the basic RNN which gives a test_accuracy less than 50%, following techniques have been experimented and a test_accuracy above 88% is achieved.

Techniques used:
- packed padded sequences
- pre-trained word embeddings
- different RNN architecture
- bidirectional RNN
- multi-layer RNN
- regularization
- a different optimizer

![sentiment](assets/images/applications/sentiment/improved.png)

### Sentiment Analysis with Attention

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Attention helps in focusing on the relevant input when predicting the sentiment of the input. Bahdanau attention was used with taking the outputs of LSTM and concatenating the final forward & backward hidden state. Without using the pre-trained word embeddings, test accuracy of `88%` is achieved.

![sentiment](assets/images/applications/sentiment/sentiment_attention_ex.png)


### Sentiment Analysis with BERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

BERT obtains new state-of-the-art results on eleven natural language processing tasks. Transfer learning in NLP has triggered after the release of BERT model. Using BERT to do the sentiment analysis is explored.

![sentiment](assets/images/applications/sentiment/bert_senti.png)


## Sentiment Analysis - HINGLISH

Mixing languages, also known as code-mixing, is a norm in multilingual societies. Multilingual people, who are non-native English speakers, tend to code-mix using English-based phonetic typing and the insertion of anglicisms in their main language.

The task is to predict the sentiment of a given code-mixed tweet. The sentiment labels are positive, negative, or neutral, and the code-mixed languages will be English-Hindi. **(Sentimix)**

Following varients have been explored:

- <a href="#sentimix-using-mlp">Sentimix using MLP</a>
- <a href="#sentimix-using-lstm">Sentimix using LSTM</a>
- <a href="#sentimix-with-xlm-roberta">Sentimix with XLM-RoBERTa</a>
- <a href="#sentimix-with-xlm-roberta-lstm-attention">Sentimix with XLM-RoBERTa-LSTM-Attention</a>
- <a href="#sentimix-with-xlm-roberta-cnn">Sentimix with XLM-RoBERTA-CNN</a>
- <a href="#sentimix-with-xlm-roberta-ensemble">Sentimix with XLM-RoBERTA-Ensemble</a>

### Sentimix using MLP

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Using the simple MLP model, `F1 score of 0.58` was achieved on test data

![codemix](assets/images/applications/sentiment/codemix_mlp.png)

### Sentimix using LSTM

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

After exploring the basic MLP model, the LSTM model was used for sentiment prediction and F1 score of `0.57` was achieved.

![codemix](assets/images/applications/sentiment/codemix_lstm.png)

The results were actually less compared to a basic MLP model. One of the reason might be LSTM is not able to learn the relationships among the words in a sentence due to the highly diverse nature of the Code-mixed data.

### Sentimix with XLM-RoBERTa

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

As the LSTM is not able to learn the relationships between the words in a code-mixed sentence due to the highly diverse nature of the Code-mixed data and no pre-trained embeddings are used, the F1 score is less.

To alleviate this problem XLM-RoBERTa model (which has been pre-trained on 100 languages) is being used to encode the sentence. In order to use the XLM-RoBERTa model, the sentence needs to be in a proper language. So first the Hinglish words need to be converted to the Hindi (Devanagari) form.

![codemix](assets/images/applications/sentiment/codemix_xlm.png)

An F1 score of `0.59` was achieved. Methods to improve this will be explored later on.

### Sentimix with XLM-RoBERTa-LSTM-Attention

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

The final output from XLM-RoBERTa model were used as input embeddings to bi-directional LSTM model. An attention layer, which takes the outputs from LSTM layer, produces a weighted representation of the input, which is then passed through a classifier for predicting the sentiment of the sentence.

![codemix](assets/images/applications/sentiment/codemix_xlm_attn.png)

An F1 score of `0.64` was achieved. 

### Sentimix with XLM-RoBERTA-CNN

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

In the same way that a 3x3 filter can look over a patch of an image, a 1x2 filter can look over a 2 sequential words in a piece of text, i.e. a bi-gram. In this CNN model we will instead use multiple filters of different sizes which will look at the bi-grams (a 1x2 filter), tri-grams (a 1x3 filter) and/or n-grams (a 1xn filter) within the text.

The intuition here is that the appearance of certain bi-grams, tri-grams and n-grams within the review will be a good indication of the final sentiment.

![codemix](assets/images/applications/sentiment/codemix_xlm_cnn.png)

An F1 score of `0.69` was achieved. 

### Sentimix with XLM-RoBERTA-Ensemble

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

CNN captures the local dependencies where as RNN captures the global dependencies. By combining both we can get better understanding of the data. Ensembling of CNN model and Bidirectional-GRU-Attention model out performs the other ones.

![codemix](assets/images/applications/sentiment/codemix_xlm_ens.png)

An F1 score of `0.71` was achieved. (Top 5 in the leaderboard).


## Document Classification

Document classification or document categorization is a problem in library science, information science and computer science. The task is to assign a document to one or more classes or categories.

Following varients have been explored:

- <a href="#document-classification-with-hierarchical-attention-network">Document Classification with Hierarchical Attention Network</a>
- <a href="#improved-han-with-regularization-techniques">Improved HAN with regularization techniques</a>


### Document Classification with Hierarchical Attention Network

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

A Hierarchical Attention Network (HAN) considers the hierarchical structure of documents (document - sentences - words) and includes an attention mechanism that is able to find the most important words and sentences in a document while taking the context into consideration.

![han](assets/images/applications/classification/han.png)


### Improved HAN with regularization techniques

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

The basic HAN model is overfitting rapidly. In order to overcome this, techniques like `Embedding Dropout`, `Locked Dropout` are explored. There is one more other technique called `Weight Dropout` which is not implemented (Let me know if there are any good resources to implement this). Pre-trained word embeddings `Glove` are also used instead of random initialization. Since the attention can be done on sentence level and word level, we can visualize which words are important in a sentence and which sentences are important in a document.

![han](assets/images/applications/classification/han_visual.png)

![han](assets/images/applications/classification/han_visual2.png)

![han](assets/images/applications/classification/han_visual3.png)

## Duplicate Question Pair Classification - Quora

QQP stands for Quora Question Pairs. The objective of the task is for a given pair of questions; we need to find whether those questions are semantically similar to each other or not.

Following varients have been explored:

- <a href="#qqp-classification-with-siamese-network">QQP Classification with Siamese Network</a>
- <a href="#qqp-classification-with-bert">QQP Classification with BERT</a>

### QQP Classification with Siamese Network

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

The algorithm needs to take the pair of questions as input and should output their similarity.
A Siamese network is used. A `Siamese neural network` (sometimes called a twin neural network) is an artificial neural network that uses the `same weights` while working in tandem on two different input vectors to compute comparable output vectors.

![qqp](assets/images/applications/classification/qqp_siamese.png)

### QQP Classification with BERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

After trying the siamese model, BERT was explored to do the Quora duplicate question pairs detection. BERT takes the question 1 and question 2 as input separated by `[SEP]` token and the classification was done using the final representation of `[CLS]` token.

![qqp](assets/images/applications/classification/qqp_bert.png)

## POS Tagging

Part-of-Speech (PoS) tagging, is a task of labelling each word in a sentence with its appropriate part of speech. 

Following varients have been explored:

- <a href="#pos-tagging-with-bilstm">POS Tagging with BiLSTM</a>
- <a href="#pos-tagging-with-transformer">POS Tagging with Transformer</a>
- <a href="#pos-tagging-with-bert">POS Tagging with BERT</a>


### POS Tagging with BiLSTM

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

This code covers the basic workflow. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop and run time (inference) tagging.

The model used is a Multi Layer Bi-directional LSTM network

![pos](assets/images/applications/classification/pos_lstm.png)

### POS Tagging with Transformer

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

After trying the RNN approach, POS tagging with Transformer based architecture is explored. Since the Transformer contains both Encoder and Decoder and for the sequence labeling task only `Encoder` will be sufficient. As the data is small having 6 layers of Encoder will overfit the data. So a 3-layer Transformer Encoder model was used.

![pos](assets/images/applications/classification/pos_transformer.png)

### POS Tagging with BERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

After trying POS tagging with Transformer Encoder, POS Tagging with pre-trained BERT model is exploed. It achieved test accuracy of `91%`.

![pos](assets/images/applications/classification/pos_bert.png)

## Natural Language Inference - SNLI

The goal of natural language inference (NLI), a widely-studied natural language processing task, is to determine if one given statement (a premise) semantically entails another given statement (a hypothesis).

Following varients have been explored:

- <a href="#nli-with-bilstm">NLI with BiLSTM</a>
- <a href="#nli-with-attention">NLI with Attention</a>
- <a href="#nli-with-transformer">NLI with Transformer</a>
- <a href="#nli-with-bert">NLI with BERT</a>
- <a href="#nli-with-distillation">NLI with Distillation</a>


### NLI with BiLSTM

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

A basic model with Siamese BiLSTM network is implemeted

![nli](assets/images/applications/classification/nli_bilstm.png)

This can be treated as base-line setup. A test accuracy of `76.84%` was achieved.

### NLI with Attention

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

In the previous notebook, the final hidden states of Premise and Hypothesis as the representations from LSTM. Now instead of taking the final hidden states, attention will be computed across all the input tokens and a final weighted vector is taken as the representation of Premise and Hypothesis. 

![nli](assets/images/applications/classification/nli_attention.png)

The test accuracy increased from `76.84%` to `79.51%`.

### NLI with Transformer

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Transformer Encoder was used to encode the Premise and Hypothesis. Once the sentence is passed through the Encoder, summation of all the tokens is considered as the final representation (others variants can be explored). The model accuracy is less compared to RNN variants.

![nli](assets/images/applications/classification/nli_transformer.png)

### NLI with BERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

NLI with Bert base model was explored. BERT takes the Premise and Hypothesis as inputs separated by `[SEP]` token and the classification was done using the final representation of `[CLS]` token.

![nli](assets/images/applications/classification/qqp_bert.png)

### NLI with Distillation

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

**`Distillation`**: A technique you can use to compress a large model, called the `teacher`, into a smaller model, called the `student`. Following student, teacher models are used in order to perform distillation on NLI.

- Student Model: Logistic Regression
- Teacher Model: Bi-directional LSTM with Attention

![nli](assets/images/applications/classification/distillation.png)


## Toxic Comment Classification

Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

Following varients have been explored:

- <a href="#toxic-comment-classification-with-gru">Toxic Comment Classification with GRU</a>
- <a href="#improved-toxic-comment-classification">Improved Toxic Comment Classification</a>
- <a href="#toxic-comment-classification-using-streamlit">Toxic Comment Classification using Streamlit</a>

### Toxic Comment Classification with GRU

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

The model used is a Bi-directional GRU network.

![toxic](assets/images/applications/classification/toxic_gru.png)

A test accuracy of `99.42%` was achieved. Since 90% of the data is not labeled into any of the toxicity, simply predicting all the data as non-toxic gives a 90% accurate model. So accuracy is not a reliable metric. A different metric ROC AUC was implemented.

### Improved Toxic Comment Classification

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

With `Categorical Cross Entropy` as the loss, roc_auc score of `0.5` is achieved. By changing the loss to `Binary Cross Entropy` and also modifying the model a bit by adding pooling layers (max, mean), the roc_auc score improved to `0.9873`.

![toxic](assets/images/applications/classification/improved_toxic.png)

### Toxic Comment Classification using Streamlit

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Converted the Toxic Comment Classification into an app using streamlit. The pre-trained model is available now.

![toxic](assets/images/applications/classification/toxic_app.png)

## Grammatically Correct Sentence - CoLA

Can artificial neural networks have the ability to judge the grammatical acceptability of a sentence? In order to explore this task, the Corpus of Linguistic Acceptability (CoLA) dataset is used. CoLA is a set of sentences labeled as grammatically correct or incorrect. 

Following varients have been explored:

- <a href="#cola-with-bert">CoLA with BERT</a>
- <a href="#cola-with-distilbert">CoLA with DistilBERT</a>

### CoLA with BERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

BERT obtains new state-of-the-art results on eleven natural language processing tasks. Transfer learning in NLP has triggered after the release of BERT model. In this notebook, we will explore how to use BERT for classifying whether a sentence is grammatically correct or not using CoLA dataset.

![cola](assets/images/applications/classification/cola_bert.png)

An accuracy of `85%` and Matthews Correlation Coefficient (MCC) of `64.1` were achieved.


### CoLA with DistilBERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

**`Distillation`**: A technique you can use to compress a large model, called the `teacher`, into a smaller model, called the `student`. Following student, teacher models are used in order to perform distillation on CoLA.

- Student Model: Distilbert base uncased
- Teacher Model: Bert base uncased

![cola](assets/images/applications/classification/distilbert.png)

Following experiments have been tried:
- Training using Bert Model (Teacher). Acc: `84.06`, MCC: `61.5`
- Training using Distilbert Model (without teacher forcing). Acc: `82.54`, MCC: `57`
- Training using Distilbert Model (with teacher forcing). Acc: `82.92`, MCC: `57.9`

## NER Tagging

Named-Entity-Recognition (NER) tagging, is a task of labelling each word in a sentence with its appropriate entity.

Following varients have been explored:

- <a href="#ner-tagging-with-bilstm">NER Tagging with BiLSTM</a>
- <a href="#ner-tagging-with-bilstm-crf">NER Tagging with BiLSTM-CRF</a>
- <a href="#ner-decoding-using-viterbi-algorithm">NER Decoding using Viterbi Algorithm</a>
- <a href="#ner-tagging-with-char-bilstm-crf">NER tagging with Char-BiLSTM-CRF</a>
- <a href="#evaluation-metrics-for-ner-tagging">Evaluation metrics for NER tagging</a>
- <a href="#ner-using-streamlit">NER using Streamlit</a>
- <a href="#ner-tagging-with-transformer">NER tagging with Transformer</a>
- <a href="#ner-tagging-with-bert">NER tagging with BERT</a>
- <a href="#ner-tagging-with-transformer-crf">NER tagging with Transformer-CRF</a>
- <a href="#ner-tagging-with-spacy">NER tagging with Spacy</a>


### NER Tagging with BiLSTM

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

This code covers the basic workflow. We'll see how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop and train, test the model.

The model used is Bi-directional LSTM network

![ner](assets/images/applications/classification/ner_lstm.png)

### NER Tagging with BiLSTM-CRF

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

In the case of Sequence tagging (NER) the tag of a current word might depend on previous word's tag. (ex: New York).

Without a CRF, we would have simply used a single linear layer to transform the output of the Bidirectional LSTM into scores for each tag. These are known as `emission scores`, which are a representation of the likelihood of the word being a certain tag.

A CRF calculates not only the emission scores but also the `transition scores`, which are the likelihood of a word being a certain tag considering the previous word was a certain tag. Therefore the transition scores measure how likely it is to transition from one tag to another.

![ner](assets/images/applications/classification/bilstm_crf.png)

### NER Decoding using Viterbi Algorithm

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

For decoding, `Viterbi` algorithm is used.

Since we're using CRFs, we're not so much predicting the right label at each word as we are predicting the right label sequence for a word sequence. Viterbi Decoding is a way to do exactly this – find the most optimal tag sequence from the scores computed by a Conditional Random Field.

![ner](assets/images/applications/classification/viterbi.png)

### NER tagging with Char-BiLSTM-CRF

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Using sub-word information in our tagging task because it can be a powerful indicator of the tags, whether they're parts of speech or entities. For example, it may learn that adjectives commonly end with "-y" or "-ul", or that places often end with "-land" or "-burg".

Therefore, our sequence tagging model uses both

- `word-level` information in the form of word embeddings.
- `character-level` information up to and including each word in both directions.

![ner](assets/images/applications/classification/char_bilstm_ner.png)

### Evaluation metrics for NER tagging

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Micro and macro-averages (for whatever metric) will compute slightly different things, and thus their interpretation differs. A macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally), whereas a micro-average will aggregate the contributions of all classes to compute the average metric. In a multi-class classification setup, micro-average is preferable if you suspect there might be class imbalance (i.e you may have many more examples of one class than of other classes).

![ner](assets/images/applications/classification/bilstm_crf_res.png)

![ner](assets/images/applications/classification/char_bilstm_crf_res.png)

### NER using Streamlit

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Converted the NER tagging into an app using streamlit. The pre-trained model(Char-BiLSTM-CRF) is available now.

![ner](assets/images/applications/classification/ner_app.png)

### NER tagging with Transformer

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

After trying the RNN approach, NER tagging with Transformer based architecture is explored. Since the Transformer contains both Encoder and Decoder and for the sequence labeling task only `Encoder` will be sufficient. A 3-layer Transformer Encoder model was used.

![ner](assets/images/applications/classification/ner_transformer.png)

### NER tagging with BERT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

After trying NER tagging with Transformer Encoder, NER Tagging with pre-trained `bert-base-cased` model is explored.

![ner](assets/images/applications/classification/ner_bert.png)


### NER tagging with Transformer-CRF

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

The transformer alone is not giving good results compared to BiLSTM in the NER tagging task. Augmenting the CRF layer on top of the transformer is implemented which is improving results compared to standalone Transformer.

![ner](assets/images/applications/classification/ner_transformer_crf.png)


### NER tagging with Spacy

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

SpaCy provides an exceptionally efficient statistical system for NER in python, which can assign labels to groups of tokens. It provides a default model which can recognize a wide range of named or numerical entities, which include person, organization, language, event etc.


![ner](assets/images/applications/classification/ner_spacy.png)

Apart from these default entities, spaCy also gives us the liberty to add arbitrary classes to the NER model, by training the model to update it with newer trained examples.

2 new entities called **`ACTIVITY`** and **`SERVICE`** in a specific domain data (bank) are created and trained with few training samples.

![ner](assets/images/applications/classification/ner_spacy_custom.png)


# Generation based Applications

<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#name-generation-with-lstm">Name Generation</a></td>
    <td style="padding:15px;"><a href="#machine-translation">Machine Translation</a></td>
    <td style="padding:15px;"><a href="#utterance-generation">Utterance Generation</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#image-captioning">Image Captioning</a></td>
    <td style="padding:15px;"><a href="#image-captioning---latex-equations">Image Captioning - LaTeX Equations</a></td>
    <td style="padding:15px;"><a href="#news-summarization-with-t5">News Summarization</a></td>
</tr>

<tr>
    <td style="padding:15px;"><a href="#email-subject-generation-with-t5">Email Subject Generation</a></td>
</tr>

</table>

## Name Generation with LSTM

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

A character-level LSTM language model is used. That is, we’ll give the LSTM a huge chunk of names and ask it to model the probability distribution of the next character in the sequence given a sequence of previous characters. This will then allow us to generate new name one character at a time

![name_gen](assets/images/applications/generation/name_gen.png)


## Machine Translation

Machine Translation (MT) is the task of automatically converting one natural language into another, preserving the meaning of the input text, and producing fluent text in the output language. Ideally, a source language sequence is translated into target language sequence. The task is to convert sentences from `German to English`.

Following varients have been explored:

- <a href="#basic-machine-translation">Basic Machine Translation</a>
- <a href="#improved-machine-translation">Improved Machine Translation</a>
- <a href="#machine-translation-with-bahdanau-attention">Machine Translation with Bahdanau Attention</a>
- <a href="#masking-packing-padded-inputs-attention-visualization-bleu-on-mt">Masking, Packing padded inputs, Attention Visualization, BLEU on MT</a>
- <a href="#machine-translation-with-transformer">Machine Translation with Transformer</a>
- <a href="#self-attention-visualization">Self Attention Visualization</a>

### Basic Machine Translation

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

The most common sequence-to-sequence (seq2seq) models are encoder-decoder models, which commonly use a recurrent neural network (RNN) to encode the source (input) sentence into a single vector. In this notebook, we'll refer to this single vector as a context vector. We can think of the context vector as being an abstract representation of the entire input sentence. This vector is then decoded by a second RNN which learns to output the target (output) sentence by generating it one word at a time.

![basic_mt](assets/images/applications/generation/basic_translation.png)


### Improved Machine Translation

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

After trying the basic machine translation which has text perplexity `36.68`, following techniques have been experimented and a test perplexity `7.041`.

- GRU is used instead of LSTM
- Single layer
- Context vector is sent to decoder rnn along with decoder input embedding
- Context vector is sent to classifier along with the decoder hidden state

![improved_mt](assets/images/applications/generation/improved_mt.png)

Checkout the code in `applications/generation` folder

### Machine Translation with Bahdanau Attention

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

The attention mechanism was born to help memorize long source sentences in neural machine translation (NMT). Rather than building a single context vector out of the encoder's last hidden state, attention is used to focus more on the relevant parts of the input while decoding a sentence. The context vector will be created by taking encoder outputs and the `previous hidden state` of the decoder rnn.

![bahdanau_mt](assets/images/applications/generation/mt_bahdanau.png)

### Masking, Packing padded inputs, Attention Visualization, BLEU on MT

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Enhancements like masking (ignoring the attention over padded input), packing padded sequences (for better computation), attention visualization and BLEU metric on test data are implemented.

![mt_visual](assets/images/applications/generation/mt_attn_visual_1.png)

### Machine Translation with Transformer

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output is used to do Machine translation from German to English

![mt_visual](assets/images/applications/generation/transformer.gif)

### Self Attention Visualization

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Run time translation (Inference) and attention visualization are added for the transformer based machine translation model.

![mt_visual](assets/images/applications/generation/attn_visual.png)

## Utterance Generation

Utterance generation is an important problem in NLP, especially in question answering, information retrieval, information extraction, conversation systems, to name a few. It could also be used to create synthentic training data for many NLP problems.

Following varients have been explored:

- <a href="#basic-utterance-generation">Basic Utterance Generation</a>
- <a href="#utterance-generation-with-attention">Utterance Generation with Attention</a>
- <a href="#utterance-generation-with-beam-search">Utterance Generation with Beam Search</a>
- <a href="#utterance-generation-with-coverage">Utterance Generation with Coverage</a>
- <a href="#utterance-generation-with-transformer">Utterance Generation with Transformer</a>
- <a href="#beam-search-in-utterance-generation-with-transformer">Beam Search in Utterance Generation with Transformer</a>
- <a href="#utterance-generation-with-bpe-tokenization">Utterance Generation with BPE Tokenization</a>
- <a href="#utterance-generation-using-streamlit">Utterance Generation using Streamlit</a>
- <a href="#general-utterance-generation">General Utterance Generation</a>


### Basic Utterance Generation

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

The most common used model for this kind of application is sequence-to-sequence network. A basic 2 layer LSTM was used.

![utt_gen](assets/images/applications/generation/basic_utterance_gen.png)

### Utterance Generation with Attention

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

The attention mechanism will help in memorizing long sentences. Rather than building a single context vector out of the encoder's last hidden state, attention is used to focus more on the relevant parts of the input while decoding a sentence. The context vector will be created by taking encoder outputs and the `hidden state` of the decoder rnn.

After trying the basic LSTM apporach, Utterance generation with attention mechanism was implemented. Inference (run time generation) was also implemented.

![utt_gen](assets/images/applications/generation/utterance_gen_attn.png)

While generating the a word in the utterance, decoder will attend over encoder inputs to find the most relevant word. This process can be visualized.

![utt_gen](assets/images/applications/generation/utt_attn_visual_1.png)


### Utterance Generation with Beam Search

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

One of the ways to mitigate the repetition in the generation of utterances is to use Beam Search. By choosing the top-scored word at each step (greedy) may lead to a sub-optimal solution but by choosing a lower scored word that may reach an optimal solution.

Instead of greedily choosing the most likely next step as the sequence is constructed, the beam search expands all possible next steps and keeps the k most likely, where k is a user-specified parameter and controls the number of beams or parallel searches through the sequence of probabilities.

![utt_gen](assets/images/applications/generation/beam_search.png)

### Utterance Generation with Coverage

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Repetition is a common problem for sequenceto-sequence models, and is especially pronounced when generating a multi-sentence text. In coverage model, we maintain a
coverage vector `c^t`, which is the sum of attention distributions over all previous decoder timesteps

![utt_gen](assets/images/applications/generation/coverage.png)

This ensures that the attention mechanism’s current decision (choosing where to attend next) is informed by a reminder of its previous decisions (summarized in c^t). This should make it easier for the attention mechanism to avoid repeatedly attending to the same locations, and thus avoid generating repetitive text.

### Utterance Generation with Transformer

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output is used to do generate utterance from a given sentence. The training time was also lot faster `4x` times compared to RNN based architecture.

![utt_gen](assets/images/applications/generation/utt_gen_transformer.png)

### Beam Search in Utterance Generation with Transformer

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Added beam search to utterance generation with transformers. With beam search, the generated utterances are more diverse and can be more than 1 (which is the case of the greedy approach). This implemented was better than naive one implemented previously.

![utt_gen](assets/images/applications/generation/utt_gen_beam.png)

### Utterance Generation with BPE Tokenization

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Utterance generation using BPE tokenization instead of Spacy is implemented.

Today, subword tokenization schemes inspired by BPE have become the norm in most advanced models including the very popular family of contextual language models like BERT, GPT-2,RoBERTa, etc.

BPE brings the perfect balance between character and word-level hybrid representations which makes it capable of managing large corpora. This behavior also enables the encoding of any rare words in the vocabulary with appropriate subword tokens without introducing any “unknown” tokens.

![utt_gen](assets/images/applications/generation/utt_gen_bpe.png)

### Utterance Generation using Streamlit

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Converted the Utterance Generation into an app using streamlit. The pre-trained model trained on the Quora dataset is available now.

![utt_gen](assets/images/applications/generation/utt_gen_app.png)

### General Utterance Generation

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Till now the Utterance Generation is trained using the `Quora Question Pairs` dataset, which contains sentences in the form of questions. When given a normal sentence (which is not in a question format) the generated utterances are very poor. This is due the `bias` induced by the dataset. Since the model is only trained on question type sentences, it fails to generate utterances in case of normal sentences. In order to generate utterances for a normal sentence, `COCO` dataset is used to train the model.
![utt_gen](assets/images/applications/generation/utt_gen_bias.png)

![utt_gen](assets/images/applications/generation/utt_gen_gen.png)


## Image Captioning

Image Captioning is the process of generating a textual description of an image. It uses both Natural Language Processing and Computer Vision techniques to generate the captions.

Flickr8K dataset is used. It contains 8092 images, each image having 5 captions.

Following varients have been explored:

- <a href="#basic-image-captioning">Basic Image Captioning</a>
- <a href="#image-captioning-with-attention">Image Captioning with Attention</a>
- <a href="#image-captioning-with-beam-search">Image Captioning with Beam Search</a>
- <a href="#image-captioning-with-bpe-tokenization">Image Captioning with BPE Tokenization</a>

### Basic Image Captioning

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). The decoder is a recurrent neural network(RNN) which takes in the encoded image and generates the caption.

In this notebook, the resnet-152 model pretrained on the ILSVRC-2012-CLS image classification dataset is used as the encoder. The decoder is a long short-term memory (LSTM) network.

![img_cap](assets/images/applications/generation/basic_image_captioning.png)

### Image Captioning with Attention

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

In this notebook, the resnet-101 model pretrained on the ILSVRC-2012-CLS image classification dataset is used as the encoder. The decoder is a long short-term memory (LSTM) network. Attention is implemented. Instead of the simple average, we use the weighted average across all pixels, with the weights of the important pixels being greater. This weighted representation of the image can be concatenated with the previously generated word at each step to generate the next word of the caption.

![img_cap](assets/images/applications/generation/img_cap_attn.png)


### Image Captioning with Beam Search

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Instead of greedily choosing the most likely next step as the caption is constructed, the beam search expands all possible next steps and keeps the k most likely, where k is a user-specified parameter and controls the number of beams or parallel searches through the sequence of probabilities.

![img_cap](assets/images/applications/generation/img_cap_beam.png)

### Image Captioning with BPE Tokenization

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Today, subword tokenization schemes inspired by BPE have become the norm in most advanced models including the very popular family of contextual language models like BERT, GPT-2,RoBERTa, etc.

BPE brings the perfect balance between character and word-level hybrid representations which makes it capable of managing large corpora. This behavior also enables the encoding of any rare words in the vocabulary with appropriate subword tokens without introducing any “unknown” tokens.

BPE was used in order to tokenize the captions instead of using nltk.

![img_cap](assets/images/applications/generation/utt_gen_bpe.png)

## Image Captioning - LaTex Equations

An application of image captioning is to convert the the equation present in the image to latex format. 

Following varients have been explored:

- <a href="#basic-image-to-latex">Basic Image to Latex</a>
- <a href="#image-to-latex-with-attention">Image to Latex with Attention</a>
- <a href="#image-to-latex-with-beam-search">Image to Latex with Beam Search</a>
- <a href="#image-to-latex-conversion-using-streamlit">Image to LaTex Conversion using Streamlit</a>

### Basic Image to Latex

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

An application of image captioning is to convert the the equation present in the image to latex format. Basic Sequence-to-Sequence models is used. CNN is used as encoder and RNN as decoder. Im2latex dataset is used. It contains 100K samples comprising of training, validation and test splits. 

![img_cap](assets/images/applications/generation/im2latex.png)

Generated formulas are not great. Following notebooks will explore techniques to improve it.

### Image to Latex with Attention

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Latex code generation using the attention mechanism is implemented. Instead of the simple average, we use the weighted average across all pixels, with the weights of the important pixels being greater. This weighted representation of the image can be concatenated with the previously generated word at each step to generate the next word of the formula.

![img_cap](assets/images/applications/generation/imgtolatex_attn.png)

### Image to Latex with Beam Search

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Added beam search in the decoding process. Also added Positional encoding to the input image and learning rate scheduler.

### Image to LaTex Conversion using Streamlit

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

Converted the Latex formula generation into an app using streamlit.

![latex](assets/images/applications/generation/latex_app.png)


## News Summarization with T5

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Automatic text summarization is the task of producing a concise and fluent summary while preserving key information content and overall meaning. Have you come across the mobile app `inshorts`? It’s an innovative news app that converts news articles into a 60-word summary.  And that is exactly what we are going to do in this notebook. The model used for this task is `T5`.

![news_sum](assets/images/applications/generation/t5_summ.png)

## Email Subject Generation with T5.

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Given the overwhelming number of emails, an effective subject line becomes essential to better inform the recipient of the email's content.

Email subject generation using T5 model was explored. AESLC dataset was used for this purpose.

![email_sub](assets/images/applications/generation/email_sub.png)


<details>
    <summary>Question-Answering based Applications</summary>
    <ul>
        <li>
            <details>
                <summary><a href="#">Basic Question Answering</a></summary>
                <ul>
                    <li><a href="#">Basic Question Answering with Dynamic Memory Network</a></li>
                    <li><a href="#">Basic Question Answering with Dynamic Memory Network Plus</a></li>
                </ul>
            </details>
        </li>
        <li>
            <details>
                <summary><a href="#">Visual Question Answering</a></summary>
                <ul>
                    <li><a href="#">Basic Visual Question Answering</a></li>
                    <li><a href="#">Visual Question Answering with Dynamic Memory Network Plus</a></li>
                </ul>
            </details>
        </li>
        <li><a href="#">Boolean Question Answering</a></li>
        <li>
            <details>
                <summary><a href="#">Closed Question Answering - SQuAD</a></summary>
                <ul>
                    <li><a href="#">Question Answering using Dynamic Co-Attention Network</a></li>
                    <li><a href="#">Question Answering using Double Cross Attention</a></li>
                </ul>
            </details>
        </li>
    </ul>
</details>



# Clustering based Applications

<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#topic-identification-in-news">Topic Identification in News</a></td>
    <td style="padding:15px;"><a href="#covid-article-finding-using-lda">Covid Article finding</a></td>
</tr>
</table>

## Topic Identification in News

Topic Identification is a Natural Language Processing (NLP) is the task to automatically extract meaning from texts by identifying recurrent themes or topics.

Following varients have been explored:

- <a href="#topic-identification-in-news-using-lda">Topic Identification in News using LDA</a>
- <a href="#improved-topic-identification-in-news-using-lda">Improved Topic Identification in News using LDA</a>
- <a href="#topic-identification-in-news-using-lsa">Topic Identification in News using LSA</a>

### Topic Identification in News using LDA

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.

Once you provide the algorithm with the number of topics, all it does it to rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.

20 Newsgroup dataset was used and only the articles are provided to identify the topics. Topic Modelling algorithms will provide for each topic what are the important words. It is upto us to infer the topic name.

![lda](assets/images/applications/clustering/lda_vis.png)

### Improved Topic Identification in News using LDA

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Choosing the number of topics is a difficult job in Topic Modelling. In order to choose the optimal number of topics, grid search is performed on various hypermeters. In order to choose the best model the model having the best perplexity score is choosed.

A good topic model will have non-overlapping, fairly big sized blobs for each topic. 

![lda](assets/images/applications/clustering/lda_imp.png)

- LDA using scikit-learn is implemented.
- Inference (predicting the topic of a given sentence) is also implemented.

### Topic Identification in News using LSA

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

We would clearly expect that the words that appear most frequently in one topic would appear less frequently in the other - otherwise that word wouldn't make a good choice to separate out the two topics. Therefore, we expect the topics to be `orthogonal`.

Latent Semantic Analysis (LSA) uses SVD. You will sometimes hear topic modelling referred to as LSA.

The SVD algorithm factorizes a matrix into one matrix with `orthogonal columns` and one with `orthogonal rows` (along with a diagonal matrix, which contains the relative importance of each factor).

![svd](assets/images/applications/clustering/svd.png)

Notes:
- SVD is a determined dimension reduction algorithm
- LDA is a probability-based generative model

## Covid article finding using LDA

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Finding the relevant article from a covid-19 research article [corpus of 50K+ documents](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) using LDA is explored.

The documents are first clustered into different topics using LDA. For a given query, dominant topic will be found using the trained LDA. Once the topic is found, most relevant articles will be fetched using the `jensenshannon` distance.

Only abstracts are used for the LDA model training. LDA model was trained using 35 topics.

![lda](assets/images/applications/clustering/covid_lda.png)


# Question Answering based Applications

<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#factual-question-answering">Factual Question Answering</a></td>
    <td style="padding:15px;"><a href="#visual-question-answering">Visual Question Answering</a></td>
    <td style="padding:15px;"><a href="#boolean-question-answering">Boolean Question Answering</a></td>
</tr>
<tr>
    <td style="padding:15px;"><a href="#closed-question-answering">Closed Question Answering</a></td>
</tr>
</table>

## Factual Question Answering

Given a set of facts, question concering them needs to be answered. Dataset used is bAbI which has 20 tasks with an amalgamation of inputs, queries and answers. See the following figure for sample.

![babi](assets/images/applications/question-answering/babi.png)

Following varients have been explored:

- <a href="#basic-question-answering-with-dynamic-memory-networks">Basic Question Answering with Dynamic Memory Networks</a>
- <a href="#question-answering-using-dmn-plus">Question Answering using DMN Plus</a>


### Basic Question Answering with Dynamic Memory Networks

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Dynamic Memory Network (DMN) is a neural network architecture which processes input sequences and questions, forms episodic memories, and generates relevant answers.

![dmn](assets/images/applications/question-answering/dmn.png)


### Question Answering using DMN Plus

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

The main difference between DMN+ and DMN is the improved InputModule for calculating the facts from input sentences keeping in mind the exchange of information between input sentences using a Bidirectional GRU and a improved version of MemoryModule using Attention based GRU model.

![dmn](assets/images/applications/question-answering/dmn_plus.png)

## Visual Question Answering

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Visual Question Answering (VQA) is the task of given an image and a natural
language question about the image, the task is to provide an accurate natural language answer.

Following varients have been explored:

- <a href="#basic-visual-question-answering">Basic Visual Question Answering</a>
- <a href="#visual-question-answering-with-dmn-plus">Visual Question Answering with DMN Plus</a>


### Basic Visual Question Answering

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

![vqa](assets/images/applications/question-answering/basic_vqa.png)

The model uses a two layer LSTM to encode the questions and the last hidden layer of VGGNet to encode the images. The image features are then l_2 normalized. Both the question and image features are transformed to a common space and fused via element-wise multiplication, which is then passed through a fully connected layer followed by a softmax layer to obtain a distribution over answers.

### Visual Question Answering with DMN Plus

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

To apply the DMN to visual question answering, input module is modified for images. The module splits an image into small local regions and considers each region equivalent to a sentence in the input module for text.

The input module for VQA is composed of three parts, illustrated in below fig: 
- local region feature extraction
- visual feature embedding
- input fusion layer

![vqa](assets/images/applications/question-answering/vqa_dmn_plus.png)

## Boolean Question Answering

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

Boolean question answering is to answer whether the question has answer present in the given context or not. The BoolQ dataset contains  the queries for complex, non-factoid information, and require difficult entailment-like inference to solve. 

![boolqa](assets/images/applications/question-answering/bool_qa.png)

## Closed Question Answering

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage

Following varients have been explored:

- <a href="#question-answering-using-dynamic-coattention-network">Question Answering using Dynamic-CoAttention-Network</a>
- <a href="#question-answering-using-double-cross-attention">Question Answering using Double-Cross-Attention</a>


### Question Answering using Dynamic-CoAttention-Network

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=hard&color=red"/>
<br/>
<br/>

The DCN first fuses co-dependent representations of the question and the document in order to focus on relevant parts of both. Then a dynamic pointing decoder iterates over potential answer spans. This iterative procedure enables the model to recover from initial local maxima corresponding to incorrect answers.

The Dynamic Coattention Network has two major parts: a coattention encoder and a
dynamic decoder. 

**`CoAttention Encoder`**: The model first encodes the
given document and question separately via the document and question encoder. The
document and question encoders are essentially a one-directional LSTM network with one
layer. Then it passes both the document and question encodings to another encoder which
computes the `coattention` via matrix multiplications and outputs the coattention encoding
from another bidirectional LSTM network.

**`Dynamic Decoder`**: Dynamic decoder is also a one-directional
LSTM network with one layer. The model runs the LSTM network through `several
iterations`. In each iteration, the LSTM takes in the final hidden state of the LSTM and the
start and end word embeddings of the answer in the last iteration and outputs a new hidden
state. Then, the model uses a `Highway Maxout Network` (HMN) to compute the new start
and end word embeddings of the answer in each iteration.

![dcn](assets/images/applications/question-answering/dcn.png)

### Question Answering using Double-Cross-Attention

Double Cross Attention (DCA) seems to provide better results compared to both BiDAF and Dynamic Co-Attention Network (DCN). The motivation behind this approach is that first we pay attention to each context and question and then we attend those attentions with respect to each other in a slightly similar way as DCN. The intuition is that if iteratively read/attend both context and question, it should help us to search for answers easily. 

I have augmented the Dynamic Decoder part from DCN model in-order to have iterative decoding process which helps finding better answer. 

![dca](assets/images/applications/question-answering/dca.png)


# Ranking Based Applications

<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#covid-19-browser">Covid-19 Browser</a></td>
</tr>

</table>


## Covid-19 Browser

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=medium&color=orange"/>
<br/>
<br/>

There was a kaggle problem on [covid-19 research challenge](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) which has over `1,00,000 +` documents. This freely available dataset is provided to the global research community to apply recent advances in natural language processing and other AI techniques to generate new insights in support of the ongoing fight against this infectious disease. There is a growing urgency for these approaches because of the rapid acceleration in new coronavirus literature, making it difficult for the medical research community to keep up.

The procedure I have taken is to convert the `abstracts` into a embedding representation using [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers/). When a query is asked, it will converted into an embedding and then ranked across the abstracts using `cosine` similarity.

![covid](assets/images/applications/ranking/covid.png)


# Recommendation based Applications

<table border="0" align="center">
<tr>
    <td style="padding:15px;"><a href="#song-recommendation">Song Recommendation</a></td>
</tr>

</table>

## Song Recommendation

<img src="https://img.shields.io/static/v1.svg?style=for-the-badge&label=difficulty&message=easy&color=green"/>
<br/>
<br/>

By taking user’s listening queue as a sentence, with each word in that sentence being a song that the user has listened to, training the Word2vec model on those sentences essentially means that for each song the user has listened to in the past, we’re using the songs they have listened to before and after to teach our model that those songs somehow belong to the same context.

![song_recom](assets/images/embeddings/song_recommendation.png)

What’s interesting about those vectors is that similar songs will have weights that are closer together than songs that are unrelated.
