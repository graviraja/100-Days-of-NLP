<h1 align="center" style="font-size:80px">
    Question Answering based Applications in NLP
</h1>

There are many question answering based problems in NLP like open domain question answering, reading comprehension, visual question answering, and others. Here I will be exploring a few of the applications.

**Note: Please raise an issue for any suggestions, corrections, and feedback.**

# Textual Question Answering

In this class, we will look into textual based question answering applications.

## Basic Question Answering with Dynamic Memory Networks.ipynb

Dynamic Memory Network (DMN) is a neural network architecture which processes input sequences and questions, forms episodic memories, and generates relevant answers.

![dmn](../../assets/images/applications/question-answering/dmn.png)


- **`Input Module`**: The Input Module processes the facts by using a GRU. For each fact the final hidden state represent the encoded fact (which will be used in Episodic Module).

- **`Question Module`**: The Question Module processes the question word by word and outputs a vector using the same GRU.

- **`Episodic Memory Module`**: The Episodic Module receives the fact and question vectors and computes which facts are more relevant to the given question. It will iterate over the facts multiple times (episodes) and udpates the memory in each iteration.

- **`Answer Module`**: By taking the memory and encoded question as inputs, Answer Module generates the answer.

Dataset used is bAbI which has 20 tasks with an amalgamation of inputs, queries and answers. See the following figure for sample.

![babi](../../assets/images/applications/question-answering/babi.png)

#### Resources

- [DMN Paper](https://arxiv.org/pdf/1506.07285.pdf)
- [Stanford Lecture on DMN](https://www.youtube.com/watch?v=T3octNTE7Is)
- [bAbI Dataset](https://research.fb.com/downloads/babi/)
- [Code reference](https://github.com/DSKSD/DeepNLP-models-Pytorch)


### Question Answering using DMN Plus.ipynb

The main difference between DMN+ and DMN is the improved InputModule for calculating the facts from input sentences keeping in mind the exchange of information between input sentences using a Bidirectional GRU and a improved version of MemoryModule using Attention based GRU model.

![dmn](../../assets/images/applications/question-answering/dmn_plus.png)

**`Input Module Improvement`**: The single GRU in DMN is replaced by two different components.

- `Positional Encoding`: The first component is a sentence reader which encodes the words in a sentence into sentence encoding using a specific encoding scheme called Positional Encoder.
- `Input Fusion Layer`: The main function of this layer is to allow the interaction between different input sentences to exchange information not only in the forward direction but also in the backward direction i.e., information from future states flowing backwards using a Bidirectional GRU module. Basically input fusion layer allows for distant supporting sentences to have a more direct interaction.

**`Memory Module Improvement`**: Attention based GRU is a modification of the original GRU by embedding information from the attention mechanism. In the GRU module, the update gate decides how much of each dimension of the hidden state to retain and how much to update depending upon the input(xi). Since update gate (ui) is calculated using only input(xi) and previous hidden state(hi_1), it certainly lacks any sort of knowledge from the question or the previous memory state. We can use these two to update the hidden state by replacing ui in GRU equation with the gate value(gi_t).

#### References
- [DMN+ Paper](https://arxiv.org/pdf/1603.01417.pdf)
- [Reference Code](https://github.com/dandelin/Dynamic-memory-networks-plus-Pytorch/)