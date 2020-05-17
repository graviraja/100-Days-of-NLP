import os
import sentencepiece as spm

DATAFILE = '../data/pg16457.txt'
MODELDIR = 'models'

spm.SentencePieceTrainer.train(f'''\
    --model_type=bpe\
    --input={DATAFILE}\
    --model_prefix={MODELDIR}/bpe\
    --vocab_size=500''')

sp = spm.SentencePieceProcessor()
sp.load(os.path.join(MODELDIR, 'bpe.model'))

# encode: text => id
print(sp.encode_as_pieces('This is a test'))
print(sp.encode_as_ids('This is a test'))

# decode: id => text
print(sp.decode_pieces(['▁T', 'h', 'is', '▁is', '▁a', '▁t', 'est']))
print(sp.decode_ids([72, 435, 26, 101, 5, 3, 153]))
