<h1 align="center" style="font-size:80px">
    Word Embeddings
</h1>

**Note: This is not a comprehensive list of tokenization methods. There may be even more ways to do the tokenization process, I am providing the most generally used methods. Please feel free to provide feedback (or) suggesting other ways.**

There are a lot of online material available to explain the concept about Word Embeddings. I can't explain any better than that. So my focus here will be on, how to use the various word embeddings in code. For each concept, I will provide the abstract and relevant resources to look into more details.


### 1. Word2Vec.ipynb: How to use Word2Vec Embeddings.

Word2Vec is one of the most popular pretrained word embeddings developed by Google. Word2Vec is trained on the Google News dataset (about 100 billion words).

The architecture of Word2Vec is really simple. It’s a feed-forward neural network with just one hidden layer. Hence, it is sometimes referred to as a Shallow Neural Network architecture.

![word2vec arch](../assets/images/embeddings/gensim.png)


Depending on the way the embeddings are learned, Word2Vec is classified into two approaches:

- Continuous Bag-of-Words (CBOW)
- Skip-gram model

Continuous Bag-of-Words (CBOW) model learns the focus word given the neighboring words whereas the Skip-gram model learns the neighboring words given the focus word. 

![word2vec arch](../assets/images/embeddings/word2vec.png)

Resources:
- [Word Embeddings - Sebastian Ruder](https://ruder.io/word-embeddings-1/)
- [Skip Gram Model - Chris McCormick](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [Learning Word Embeddings Andrew NG](https://www.youtube.com/watch?v=xtPXjvwCt64)
- [Word2Vec Andrew NG](https://www.youtube.com/watch?v=jak0sKPoKu8)
- [Stanford NLP Lecture 1](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=1)
- [Word2Vec Paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [Google Word2Vec](https://code.google.com/archive/p/word2vec/)
