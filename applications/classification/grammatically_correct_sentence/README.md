# Grammatically Correct Sentence

Can artificial neural networks have the ability to judge the grammatical acceptability of a sentence? In order to explore this task, the [Corpus of Linguistic Acceptability (CoLA)](https://nyu-mll.github.io/CoLA/) dataset is used. CoLA is a set of sentences labeled as grammatically correct or incorrect. 

### CoLA with BERT.ipynb

BERT obtains new state-of-the-art results on eleven natural language processing tasks. Transfer learning in NLP has triggered after the release of BERT model. In this notebook, we will explore how to use BERT for classifying whether a sentence is grammatically correct or not using CoLA dataset.

Classifying whether a sentence is grammatically correct or not using BERT can be summarized as:
- Tokenize the sentence using BERT tokenizer (word-piece)
- Add the specical tokens used in BERT
- Send the tokenized sentence through BERT layers
- Take the final hidden state of the `[CLS]` token
- Send this hidden state to through a simple linear classifier which predicts the grammatical correctness.

![cola](../../../assets/images/applications/classification/cola_bert.png)

An accuracy of `85%` and Matthews Correlation Coefficient (MCC) of `64.1` were achieved.

#### Resources

- [Medium blog on MCC](https://towardsdatascience.com/the-best-classification-metric-youve-never-heard-of-the-matthews-correlation-coefficient-3bf50a2f3e9a)


### CoLA with DistilBERT.ipynb

**`Distillation`**: A technique you can use to compress a large model, called the `teacher`, into a smaller model, called the `student`. Following student, teacher models are used in order to perform distillation on CoLA.

- Student Model: Distilbert base uncased
- Teacher Model: Bert base uncased

![cola](../../../assets/images/applications/classification/distilbert.png)

Following experiments have been tried:
- Training using Bert Model (Teacher). Acc: `84.06`, MCC: `61.5`
- Training using Distilbert Model (without teacher forcing). Acc: `82.54`, MCC: `57`
- Training using Distilbert Model (with teacher forcing). Acc: `82.92`, MCC: `57.9`

#### Resources

- [Distillation blog by Victor Sanh (A must read)](https://medium.com/huggingface/distilbert-8cf3380435b5)