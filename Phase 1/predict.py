import pickle
from spacy.language import Language
from spacy.tokens.span import Span

def getNER() -> Language:
    return pickle.load(open("../NER", "rb")) 
ner = getNER()

def detectRelation(first: Span, second: Span, sentence: Span):
    return "bruh"