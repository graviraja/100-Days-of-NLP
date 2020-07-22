<h1 align="center" style="font-size:80px">
    Clustering based Applications in NLP
</h1>

One of the primary applications of natural language processing is to automatically extract what topics people are discussing (or) what the content is about from large volumes of text. Some examples of large text could be feeds from social media, customer reviews of hotels, movies, etc, user feedbacks, news stories, e-mails of customer complaints etc.

Knowing what people are talking about and understanding their problems and opinions is highly valuable to businesses, administrators, political campaigns. And it’s really hard to manually read through such large volumes and compile the topics.

Thus is required an automated algorithm that can read through the text documents and automatically output the topics discussed. Clustering algorithms will help in indentifying and grouping the documents based on their content.


**Note: Please raise an issue for any suggestions, corrections, and feedback.**

# Topic Modelling

Topic modeling is an unsupervised machine learning technique that’s capable of scanning a set of documents, detecting word and phrase patterns within them, and automatically clustering word groups and similar expressions that best characterize a set of documents.

## Topic Identification in News using LDA.ipynb

LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion.

Once you provide the algorithm with the number of topics, all it does it to rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.

20 Newsgroup dataset was used and only the articles are provided to identify the topics. Topic Modelling algorithms will provide for each topic what are the important words. It is upto us to infer the topic name.

## Improved Topic Identification in News using LDA.ipynb

Choosing the number of topics is a difficult job in Topic Modelling. In order to choose the optimal number of topics, grid search is performed on various hypermeters. In order to choose the best model the model having the best perplexity score is choosed.

A good topic model will have non-overlapping, fairly big sized blobs for each topic. 

- LDA using scikit-learn is implemented.
- Inference (predicting the topic of a given sentence) is also implemented.

## Improved Topic Identification in News using LSA.ipynb

We would clearly expect that the words that appear most frequently in one topic would appear less frequently in the other - otherwise that word wouldn't make a good choice to separate out the two topics. Therefore, we expect the topics to be `orthogonal`.

Latent Semantic Analysis (LSA) uses SVD. You will sometimes hear topic modelling referred to as LSA.

The SVD algorithm factorizes a matrix into one matrix with `orthogonal columns` and one with `orthogonal rows` (along with a diagonal matrix, which contains the relative importance of each factor).

Notes:
- SVD is a determined dimension reduction algorithm
- LDA is a probability-based generative model

## Covid article finding using LDA.ipynb

Finding the relevant article from a covid-19 research article [corpus of 50K+ documents](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) using LDA is explored.

The documents are first clustered into different topics using LDA. For a given query, dominant topic will be found using the trained LDA. Once the topic is found, most relevant articles will be fetched using the `jensenshannon` distance.

Only abstracts are used for the LDA model training. LDA model was trained using 35 topics.