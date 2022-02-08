import pickle
from spacy.language import Language
from spacy.tokens.span import Span

def getNER() -> Language:
    return pickle.load(open("NER", "rb")) 
ner = getNER()

def detectRelation(first: Span, second: Span, sentence: Span):
    return extractPattern(first, second, sentence)

def extractPattern(first: Span, second: Span, sentence: Span):
    #accumulator lists
    path1 = list()
    path2 = list()
    #sentence root
    root = sentence.root
    #tracing the entities up the dependency tree
    pointer1 = first.root#first
    while pointer1 != root:
        path1.append(pointer1.head.lemma_)
        pointer1 = pointer1.head
    pointer2 = second.root#second
    while pointer2 != root:
        path2.append(pointer2.head.lemma_)
        pointer2 = pointer2.head
    #remove duplicates
    i = -1
    while len(path1) >= abs(i) and len(path2) >= abs (i) and path1[i] == path2[i]:
        i -= 1
    i += 1
    pattern = path1[0:i] + path2[i::-1]
    return ".".join(pattern)

