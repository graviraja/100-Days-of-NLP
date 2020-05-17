import os
import sentencepiece as spm

DATAFILE = '../data/pg16457.txt'
MODELDIR = 'models'

spm.SentencePieceTrainer.train(f'''\
    --model_type=unigram\
    --input={DATAFILE}\
    --model_prefix={MODELDIR}/uni\
    --vocab_size=500''')

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(MODELDIR, 'uni.model'))

input_string = "This is a test"
# encode: text => id
# by default a space is added at the start of the input sentence
print(sp.encode_as_pieces(input_string))    # ['▁This', '▁is', '▁a', '▁t', 'est']
print(sp.encode_as_ids(input_string))       # [371, 77, 13, 101, 181]

# decode: id => text
print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))    # This is a test
print(sp.decode_ids([371, 77, 13, 101, 181]))                     # This is a test

# returns vocab size
print(f"vocab size: {sp.get_piece_size()}")

# id <=> piece conversion
print(f"id 371 to piece: {sp.id_to_piece(371)}")
print(f"Piece ▁This to id: {sp.piece_to_id('▁This')}")

# You can see from the code that we used the “id_to_piece” function which turns the ID of a token into its corresponding textual representation.

# This is important since SentencePiece enables the subword process to be reversible.
# You can encode your test sentence in ID’s or in subword tokens; what you use is up to you.
# The key is that you can decode either the IDs or the tokens perfectly back into the original sentences,
# including the original spaces. Previously this was not possible with other tokenizers since they just 
# provided the tokens and it was not clear exactly what encoding scheme was used,
# e.g. how did they deal with spaces or punctuation? This is a big selling point for SentencePiece.
tokens = ['▁This', '▁is', '▁a', '▁t', 'est']
merged = "".join(tokens).replace('▁', " ").strip()
assert merged == input_string, "Input string and detokenized sentence didn't match"


# Sampling
# This is only available for the Unigram model since BPE is a frequency-based approach
for n in range(10):
  print(sp.sample_encode_as_pieces('remembers', -1, 0.1))

# nbest samples
print(sp.nbest_encode_as_pieces('remembers', 10))

# <unk>, <s>, </s> are defined by default. Their ids are (0, 1, 2)
# <s> and </s> are defined as 'control' symbol.
# control symbol: We only reserve ids for these tokens. Even if these tokens appear in the input text, 
# they are not handled as one token. User needs to insert ids explicitly after encoding.
for id in range(3):
  print(sp.id_to_piece(id), sp.is_control(id))


# We can define special tokens (symbols) to tweak the DNN behavior through the tokens. Typical examples are BERT's special symbols., e.g., [SEP] and [CLS].

# There are two types of special tokens:

# user defined symbols: Always treated as one token in any context. These symbols can appear in the input sentence.
# control symbol: We only reserve ids for these tokens. Even if these tokens appear in the input text, they are not handled as one token. User needs to insert ids explicitly after encoding.

# Refer to this for more details: https://colab.research.google.com/github/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb#scrollTo=dngckiPMcWbA

# ## Example of user defined symbols
spm.SentencePieceTrainer.train(f'''\
    --model_type=unigram\
    --input={DATAFILE}\
    --model_prefix={MODELDIR}/uni_user\
    --user_defined_symbols=<sep>,<cls>\
    --vocab_size=500''')
sp_user = spm.SentencePieceProcessor()
sp_user.load(os.path.join(MODELDIR, 'uni_user.model'))


# ids are reserved in both mode.
# <unk>=0, <s>=1, </s>=2, <sep>=3, <cls>=4
# user defined symbols allow these symbol to apper in the text.
print(sp_user.encode_as_pieces('this is a test<sep> hello world<cls>'))
print(sp_user.piece_to_id('<sep>'))  # 3
print(sp_user.piece_to_id('<cls>'))  # 4
print('3=', sp_user.decode_ids([3]))  # decoded to <sep>
print('4=', sp_user.decode_ids([4]))  # decoded to <cls>

print('bos=', sp_user.bos_id())     # 1
print('eos=', sp_user.eos_id())     # 2
print('unk=', sp_user.unk_id())     # 0
print('pad=', sp_user.pad_id())     # -1, disabled by default

print(sp_user.encode_as_ids('Hello world'))     # [148, 9, 122, 11, 461]

# Prepend or append bos/eos ids.
print([sp_user.bos_id()] + sp_user.encode_as_ids('Hello world') + [sp_user.eos_id()])   # [1, 148, 9, 122, 11, 461, 2]