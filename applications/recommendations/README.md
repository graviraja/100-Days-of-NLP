<h1 align="center" style="font-size:80px">
    Recommendation based Applications in NLP
</h1>

# Recommendation systems fall under two broad categories:

- **`Content-based systems`** are recommendation systems that are based on the features of the item we’re trying to recommend. When talking about music, this includes for example the genre of the song or how many beats per minute it has.

- **`Collaborative Filtering-based systems`** are systems that rely on historical usage data to recommend items that other similar users have previously interacted with. These systems are oblivious to the features of the content itself, and base their recommendations on the principle that people who have many songs or artists in common, will generally like the same styles of music.


With enough data, collaborative filtering systems turn out to be effective at recommending relevant items. The basic idea behind collaborative filtering is that if user 1 likes artists A & B, and user 2 likes artists A, B & C, then it is likely that user 1 will also be interested in artist C.

## Song Recommendation.ipynb: How to build Song Recommendations using Word2Vec

The Word2vec Skip-gram model is a shallow neural network with a single hidden layer that takes in a word as input and tries to predict the context of words around it as output.

But how does that relate to music recommendations? Well, we can think of a user’s listening queue as a sentence, with each word in that sentence being a song that the user has listened to. So then, training the Word2vec model on those sentences essentially means that for each song the user has listened to in the past, we’re using the songs they have listened to before and after to teach our model that those songs somehow belong to the same context. Here’s an idea of what the neural network would look like with songs instead of words:

![song_recom](../../assets/images/embeddings/song_recommendation.png)

This is the same approach as the analysis of text discussed above, except instead of textual words we now have a unique identifier for each song.
What we get at the end of the training phase is a model where each song is represented by a vector of weights in a high dimensional space. What’s interesting about those vectors is that similar songs will have weights that are closer together than songs that are unrelated.

### Resources


- [Word2vec for music recommnedations](https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484)

- [Intuition & Uses-cases of Embeddings in NLP](https://www.youtube.com/watch?v=4-QoMdSqG_I)
