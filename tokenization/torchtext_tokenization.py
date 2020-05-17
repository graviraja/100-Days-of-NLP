import os
import sentencepiece as spm

import torchtext
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("spacy")
spacy_tokens = tokenizer("You can now install TorchText using pip!")
print(f"Spacy tokens: {spacy_tokens}")  # ['You', 'can', 'now', 'install', 'TorchText', 'using', 'pip', '!']


tokenizer = get_tokenizer("basic_english")
basic_english_tokens = tokenizer("You can now install TorchText using pip!")
print(f"Basic English tokens: {basic_english_tokens}") # ['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']
# note that all the tokens are converted into lowercase


tokenizer = get_tokenizer("moses")
moses_tokens = tokenizer("You can now install TorchText using pip!")
print(f"Moses tokens: {moses_tokens}")  # ['You', 'can', 'now', 'install', 'TorchText', 'using', 'pip', '!']


# custom tokenizer
# let's see how to configure sentencepiece tokenizer to torchtext

DATAFILE = '../data/pg16457.txt'
MODELDIR = 'models'

spm.SentencePieceTrainer.train(f'''\
    --model_type=bpe\
    --input={DATAFILE}\
    --model_prefix={MODELDIR}/bpe\
    --vocab_size=500''')

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(MODELDIR, 'bpe.model'))


def custom_tokenizer(sentence):
    return sp.encode_as_pieces(sentence)

# in-order to provide a custom tokenizer, it must have the functionality 
# of taking a single string and should provide the tokens for the string
tokenizer = get_tokenizer(custom_tokenizer)
sp_tokens = tokenizer("You can now install TorchText using pip!")
print(f"sp tokens: {sp_tokens}")  # ['▁', 'Y', 'ou', '▁can', '▁now', '▁in', 'st', 'all', '▁T', 'or', 'ch', 'T', 'e', 'x', 't', '▁us', 'ing', '▁p', 'i', 'p', '!']
