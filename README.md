
![emb](assets/images/embeddings/header.png)


 > There is nothing magic about magic. The magician merely understands something simple which doesn’t appear to be simple or natural to the untrained audience. Once you learn how to hold a card while making your hand look empty, you only need practice before you, too, can “do magic.” – Jeffrey Friedl in the book Mastering Regular Expressions

**Note: Please raise an issue for any suggestions, corrections, and feedback.**

<h1 align="center" style="font-size:60px">
    Building Blocks of NLP
</h1>

## Tokenization

### Day 1


The process of converting textual data to tokens, is one of the most important step in NLP. Tokenization using the following methods has been explored:

- [Spacy](https://spacy.io/usage/linguistic-features#tokenization)
- [Byte Pair Encoding (Sentencepiece)](https://github.com/google/sentencepiece)
- [Unigram Encoding (Sentencepiece)](https://github.com/google/sentencepiece)
- [Torchtext](https://pytorch.org/text/data_utils.html)
- [Tokenizers](https://github.com/huggingface/tokenizers)

Checkout the code in `tokenization` folder

## Word Embeddings

A word embedding is a learned representation for text where words that have the same meaning have a similar representation. It is this approach to representing words and documents that may be considered one of the key breakthroughs of deep learning on challenging natural language processing problems.

![emb](assets/images/embeddings/embeddings.png)

### Day 2: Word2Vec

Word2Vec is one of the most popular pretrained word embeddings developed by Google. Depending on the way the embeddings are learned, Word2Vec is classified into two approaches:

- Continuous Bag-of-Words (CBOW)
- Skip-gram model


![word2vec arch](assets/images/embeddings/word2vec.png)


### Day 3: GloVe

GloVe is another commonly used method of obtaining pre-trained embeddings. GloVe aims to achieve two goals:

- Create word vectors that capture meaning in vector space
- Takes advantage of global count statistics instead of only local information

### Day 4: ELMo

ELMo is a deep contextualized word representation that models:

- complex characteristics of word use (e.g., syntax and semantics)
- how these uses vary across linguistic contexts (i.e., to model polysemy).

These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus.

![elmo arch](assets/images/embeddings/elmo.png)


Checkout the code in `embeddings` folder

## Architectures & Techniques

There are several ways the input can be processed after tokenization. One can use different machine learning algorithms, statistical methods (or) deep learning architectures. Here I will try to cover some of the most prominent architectures & techniques used in NLP like RNN, Attention mechanism, ULMFit, Transformer, GPT-2, BERT, and others.

### Day 6: RNN

Recurrent networks - RNN, LSTM, GRU have proven to be one of the most important unit in NLP applications because of their architecture. There are many problems where the sequence nature needs to be remembered like in order to predict an emotion in the scene, previous scenes needs to be remembered.

![rnn gif](./assets/images/architectures/rnn.gif)

### Day 9: pack_padded_sequences

When training RNN (LSTM or GRU or vanilla-RNN), it is difficult to batch the variable-length sequences. Ideally we will pad all the sequences to a fixed length and end up doing un-necessary computations. How can we overcome this? PyTorch provides the `pack_padded_sequences` functionality.

![pack img](./assets/images/architectures/pack_padded_seq.jpg)

### Day 13: Luong Attention

The attention mechanism was born to help memorize long source sentences in neural machine translation (NMT). Rather than building a single context vector out of the encoder's last hidden state, attention is used to focus more on the relevant parts of the input while decoding a sentence. The context vector will be created by taking encoder outputs and the `current output` of the decoder rnn.

![pack img](./assets/images/architectures/luong_attention.png)

The attention score can be calculated in three ways. `dot`, `general` and `concat`.

![luong_fn](./assets/images/architectures/luong_fn.png)


### Day 14: Bahdanau Attention

The major difference between Bahdanau & Luong attention is the way the context vector is created. The context vector will be created by taking encoder outputs and the `previous hidden state` of the decoder rnn. Where is in Luong attention the context vector will be created by taking encoder outputs and the `current hidden state` of the decoder rnn.

Once the context is calculated it is combined with decoder input embedding and fed as input to decoder rnn.

![pack img](./assets/images/architectures/bahdanau_attention.png)

The Bahdanau attention is also called as `additive` attention.

![bahdanau_fn](./assets/images/architectures/bahdanau_fn.jpg)


### Day 18: Transformer

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. Such attention mechanisms are used in conjunction with a recurrent network.

The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

![transformer](./assets/images/architectures/transformer.png)

### Day 23: GPT-2

The GPT-2 paper states that: 

> Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when trained on a new dataset of millions of webpages called WebText. Our largest model, GPT-2, is a 1.5B parameter Transformer that achieves state of the art results on 7 out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText. Samples from the model reflect these improvements and contain coherent paragraphs of text. These findings suggest a promising path towards building language processing systems which learn to perform tasks from their naturally occurring demonstrations.

![gpt2](./assets/images/architectures/gpt2_usecase.png)

The GPT-2 utilizes a 12-layer Decoder Only Transformer architecture.

![gpt2](./assets/images/architectures/gpt2.png)


### Day 24: BERT

Language modeling is an effective task for using unlabeled data to pretrain neural networks in NLP. Traditional language models take the previous n tokens and predict the next one. In contrast, BERT trains a language model that takes both the previous and next tokens into account when predicting. BERT is also trained on a next sentence prediction task to better handle tasks that require reasoning about the relationship between two sentences (e.g. similar questions or not)

![bert](./assets/images/architectures/bert.png)

BERT uses the Transformer architecture for encoding sentences.

![bert](./assets/images/architectures/bert_arch.png)

Checkout the code in `architectures` folder

<h1 align="center" style="font-size:60px">
    Applications of NLP
</h1>

There are many kinds of NLP problems like chatbots, sentiment classification, machine translation, document classification, named entity recognition, text summarization, natural language inference, information retrieval, image captioning, emotion recognition, recommendation systems, and many others. Here, I will try to work on some of the problems in NLP.

## Recommendation based Applications

### Day 5: Song Recommendation

By taking user’s listening queue as a sentence, with each word in that sentence being a song that the user has listened to, training the Word2vec model on those sentences essentially means that for each song the user has listened to in the past, we’re using the songs they have listened to before and after to teach our model that those songs somehow belong to the same context.

![song_recom](assets/images/embeddings/song_recommendation.png)

What’s interesting about those vectors is that similar songs will have weights that are closer together than songs that are unrelated.

Checkout the code in `applications/recommendations` folder

## Classification based Applications

### Day 7: Simple Sentiment Classification with RNN - IMDB

Sentiment analysis refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information.

As an example, IMDb dataset is used and simpleRNN is used for processing and identifying the sentiment.

![sentiment](assets/images/applications/sentiment/simple.gif)

### Day 8: Improved Sentiment Classification - IMDB

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

### Day 17: Sentiment Analysis with Attention - IMDB

Attention helps in focusing on the relevant input when predicting the sentiment of the input. Bahdanau attention was used with taking the outputs of LSTM and concatenating the final forward & backward hidden state. Without using the pre-trained word embeddings, test accuracy of `88%` is achieved.

![sentiment](assets/images/applications/sentiment/sentiment_attention_ex.png)


### Day 21: Document Classification with Hierarchical Attention Network

A Hierarchical Attention Network (HAN) considers the hierarchical structure of documents (document - sentences - words) and includes an attention mechanism that is able to find the most important words and sentences in a document while taking the context into consideration.

![han](assets/images/applications/classification/han.png)


### Day 22: Improved HAN with regularization techniques

The basic HAN model is overfitting rapidly. In order to overcome this, techniques like `Embedding Dropout`, `Locked Dropout` are explored. There is one more other technique called `Weight Dropout` which is not implemented (Let me know if there are any good resources to implement this). Pre-trained word embeddings `Glove` are also used instead of random initialization. Since the attention can be done on sentence level and word level, we can visualize which words are important in a sentence and which sentences are important in a document.

![han](assets/images/applications/classification/han_visual.png)

![han](assets/images/applications/classification/han_visual2.png)

![han](assets/images/applications/classification/han_visual3.png)

Checkout the code in `applications/classification` folder

## Generation based Applications

### Day 10: Name Generation with LSTM

A character-level LSTM language model is used. That is, we’ll give the LSTM a huge chunk of names and ask it to model the probability distribution of the next character in the sequence given a sequence of previous characters. This will then allow us to generate new name one character at a time

![name_gen](assets/images/applications/generation/name_gen.png)

Checkout the code in `applications/generation` folder

### Day 11: Basic Machine Translation: German to English

The most common sequence-to-sequence (seq2seq) models are encoder-decoder models, which commonly use a recurrent neural network (RNN) to encode the source (input) sentence into a single vector. In this notebook, we'll refer to this single vector as a context vector. We can think of the context vector as being an abstract representation of the entire input sentence. This vector is then decoded by a second RNN which learns to output the target (output) sentence by generating it one word at a time.

![basic_mt](assets/images/applications/generation/basic_translation.png)

Checkout the code in `applications/generation` folder

### Day 12: Improved Machine Translation: German to English

After trying the basic machine translation which has text perplexity `36.68`, following techniques have been experimented and a test perplexity `7.041`.

- GRU is used instead of LSTM
- Single layer
- Context vector is sent to decoder rnn along with decoder input embedding
- Context vector is sent to classifier along with the decoder hidden state

![improved_mt](assets/images/applications/generation/improved_mt.png)

Checkout the code in `applications/generation` folder

### Day 15: Machine Translation with Bahdanau Attention: German to English

The attention mechanism was born to help memorize long source sentences in neural machine translation (NMT). Rather than building a single context vector out of the encoder's last hidden state, attention is used to focus more on the relevant parts of the input while decoding a sentence. The context vector will be created by taking encoder outputs and the `previous hidden state` of the decoder rnn.

![bahdanau_mt](assets/images/applications/generation/mt_bahdanau.png)

### Day 16: Masking, Packing padded inputs, Attention Visualization, BLEU on MT: German to English

Enhancements like masking (ignoring the attention over padded input), packing padded sequences (for better computation), attention visualization and BLEU metric on test data are implemented.

![mt_visual](assets/images/applications/generation/mt_attn_visual_1.png)

### Day 19: Machine Translation with Transformer: German to English

The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output is used to do Machine translation from German to English

![mt_visual](assets/images/applications/generation/transformer.gif)

### Day 20: Self Attention Visualization

Run time translation (Inference) and attention visualization are added for the transformer based machine translation model.

![mt_visual](assets/images/applications/generation/attn_visual.png)
