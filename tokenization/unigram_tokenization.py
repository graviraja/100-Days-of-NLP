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

# encode: text => id
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids('This is a test'))

# decode: id => text
print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
print(sp.decode_ids([371, 77, 13, 101, 181]))
