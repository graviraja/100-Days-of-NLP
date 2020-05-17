"""
Tokenization using spacy
"""
import spacy
from spacy.symbols import ORTH

# direct use
nlp = spacy.load("en_core_web_sm")

text = '''Apple is looking at buying "U.K." startup for $1 billion!'''
doc = nlp(text)

print("\n======= Tokens =======")
# tokens
for token in doc:
    print(token.text)

# token explaination
print("\n======= Tokenization explaination =======")
tok_exp = nlp.tokenizer.explain(text)
for t in tok_exp:
    print(t[1], "\t", t[0])

# NOTE: Detokenization without doc is difficult in spacy. 

print("\n======= Tokens information =======")
# spacy offers a lot of other information along with tokens
for token in doc:
    print(f"""token: {token.text},\
    lemmatization: {token.lemma_},\
    pos: {token.pos_},\
    is_alpha: {token.is_alpha},\
    is_stopword: {token.is_stop}""")

print("\n======= Customization =======")
# customization
doc = nlp("gimme that")  # phrase to tokenize
print([w.text for w in doc])  # ['gimme', 'that']

# Add special case rule
special_case = [{ORTH: "gim"}, {ORTH: "me"}]
nlp.tokenizer.add_special_case("gimme", special_case)

# Check new tokenization
print([w.text for w in nlp("gimme that")])  # ['gim', 'me', 'that']

# The special case rules have precedence over the punctuation splitting
doc = nlp(".....gimme!!!! that")    # phrase to tokenize
print([w.text for w in doc])    # ['.....', 'gim', 'me', '!', '!', '!', '!', 'that']
