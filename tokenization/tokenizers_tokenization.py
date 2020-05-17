from tokenizers import (BertWordPieceTokenizer,
                        SentencePieceBPETokenizer,
                        ByteLevelBPETokenizer,
                        CharBPETokenizer)

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

tokenizer = BertWordPieceTokenizer("../data/bert-base-uncased-vocab.txt", lowercase=True)
print(tokenizer)

# Tokenizers provide exhaustive outputs: tokens, mapping to original string, attention/special token masks.
# They also handle model's max input lengths as well as padding (to directly encode in padded batches)
output = tokenizer.encode("Hello, y'all! How are you?")

print(output)
print(f"ids: {output.ids}")
print(f"type_ids: {output.type_ids}")
print(f"tokens: {output.tokens}")
print(f"offsets: {output.offsets}")
print(f"attention_mask: {output.attention_mask}")
print(f"special_tokens_mask: {output.special_tokens_mask}")
print(f"overflowing: {output.overflowing}")

# Provided tokenizers
# CharBPETokenizer: The original BPE
# ByteLevelBPETokenizer: The byte level version of the BPE
# SentencePieceBPETokenizer: A BPE implementation compatible with the one used by SentencePiece
# BertWordPieceTokenizer: The famous Bert tokenizer, using WordPiece

DATAFILE = '../data/pg16457.txt'
MODELDIR = 'models'

input_text = 'This is a test'

# # CharBPETokenizer
# tokenizer = CharBPETokenizer()
# tokenizer.train([DATAFILE], vocab_size=500)

# tokenizer.save(MODELDIR, 'char_bpe')

# output = tokenizer.encode(input_text)
# print(output.tokens)

# # ByteLevelBPETokenizer
# tokenizer = ByteLevelBPETokenizer()
# tokenizer.train([DATAFILE], vocab_size=500)

# tokenizer.save(MODELDIR, 'byte_bpe')
# output = tokenizer.encode(input_text)
# print(output.tokens)

# # SentencePieceBPETokenizer
# tokenizer = ByteLevelBPETokenizer()
# tokenizer.train([DATAFILE], vocab_size=500)

# tokenizer.save(MODELDIR, 'tok_sp_bpe')
# output = tokenizer.encode(input_text)
# print(output.tokens)


# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel()

# And then train
trainer = trainers.BpeTrainer(vocab_size=500)
tokenizer.train(trainer, [
    DATAFILE
])

# Now we can encode
encoded = tokenizer.encode(input_text)
print(encoded.tokens)
