import torchtext
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("spacy")
spacy_tokens = tokenizer("You can now install TorchText using pip!")
print(spacy_tokens)

# ['You', 'can', 'now', 'install', 'TorchText', 'using', 'pip', '!']


# tokenizer = get_tokenizer("basic_english")
# basic_english_tokens = tokenizer("You can now install TorchText using pip!")
# print(basic_english_tokens)



# tokenizer = get_tokenizer("moses")
# moses_tokens = tokenizer("You can now install TorchText using pip!")
# print(moses_tokens)


# tokenizer = get_tokenizer("revtok")
# basic_english_tokens = tokenizer("You can now install TorchText using pip!")
# print(basic_english_tokens)


# tokenizer = get_tokenizer("basic_english")
# basic_english_tokens = tokenizer("You can now install TorchText using pip!")
# print(basic_english_tokens)