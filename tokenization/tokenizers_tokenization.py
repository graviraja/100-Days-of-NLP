from tokenizers import (BertWordPieceTokenizer,
                        SentencePieceBPETokenizer,
                        ByteLevelBPETokenizer,
                        CharBPETokenizer)

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

tokenizer = BertWordPieceTokenizer("../data/bert-base-uncased-vocab.txt", lowercase=True)
print(tokenizer)
# Tokenizer(vocabulary_size=30522, model=BertWordPiece, unk_token=[UNK],
# sep_token=[SEP], cls_token=[CLS], pad_token=[PAD], mask_token=[MASK],
# clean_text=True, handle_chinese_chars=True, strip_accents=True,
# lowercase=True, wordpieces_prefix=##)

# Tokenizers provide exhaustive outputs: tokens, mapping to original string, attention/special token masks.
# They also handle model's max input lengths as well as padding (to directly encode in padded batches)
output = tokenizer.encode("Hello, y'all! How are you?")

print(output)   # Encoding(num_tokens=12, attributes=[ids, type_ids, tokens, offsets, attention_mask, 							 special_tokens_mask, overflowing])
print(f"ids: {output.ids}") # [101, 7592, 1010, 1061, 1005, 2035, 999, 2129, 2024, 2017, 1029, 102]
print(f"type_ids: {output.type_ids}")   # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(f"tokens: {output.tokens}")   # ['[CLS]', 'hello', ',', 'y', "'", 'all', '!', 'how', 'are', 											'you', '?', '[SEP]']
print(f"offsets: {output.offsets}") # [(0, 0), (0, 5), (5, 6), (7, 8), (8, 9), (9, 12), (12, 13), 
                                    #  (14,17), (18, 21), (22, 25), (25, 26), (0, 0)]
print(f"attention_mask: {output.attention_mask}")   # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
print(f"special_tokens_mask: {output.special_tokens_mask}") # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
print(f"overflowing: {output.overflowing}") # []

# Provided tokenizers
# CharBPETokenizer: The original BPE
# ByteLevelBPETokenizer: The byte level version of the BPE
# SentencePieceBPETokenizer: A BPE implementation compatible with the one used by SentencePiece
# BertWordPieceTokenizer: The famous Bert tokenizer, using WordPiece

DATAFILE = '../data/pg16457.txt'
MODELDIR = 'models'

input_text = 'This is a test'

# Training the tokenizers

print("========= CharBPETokenizer ==========")
# CharBPETokenizer
tokenizer = CharBPETokenizer()
tokenizer.train([DATAFILE], vocab_size=500)

tokenizer.save(MODELDIR, 'char_bpe')

output = tokenizer.encode(input_text)
print(output.tokens)    # ['T', 'his</w>', 'is</w>', 'a</w>', 't', 'est</w>']

print("========= ByteLevelBPETokenizer ==========")
# ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train([DATAFILE], vocab_size=500)

tokenizer.save(MODELDIR, 'byte_bpe')
output = tokenizer.encode(input_text)
print(output.tokens)    # ['T', 'h', 'is', 'Ġis', 'Ġa', 'Ġt', 'est']

print("========= SentencePieceBPETokenizer ==========")
# SentencePieceBPETokenizer
tokenizer = SentencePieceBPETokenizer()
tokenizer.train([DATAFILE], vocab_size=500)

tokenizer.save(MODELDIR, 'tok_sp_bpe')
output = tokenizer.encode(input_text)
print(output.tokens)    # ['▁T', 'h', 'is', '▁is', '▁a', '▁t', 'est']

print("========= BertWordPieceTokenizer ==========")
# BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer()
tokenizer.train([DATAFILE], vocab_size=500)

tokenizer.save(MODELDIR, 'bert_bpe')
output = tokenizer.encode(input_text)
print(output.tokens)    # ['this', 'is', 'a', 't', '##est']
