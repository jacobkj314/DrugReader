import pickle
from spacy.language import Language
from spacy.tokens.span import Span

ner: Language = pickle.load(open("NER", "rb")) 

labels = ["effect", "mechanism", "advise", "int"]
goldPatterns: dict[str, set[set]] = pickle.load(open("goldPatterns", "rb"))

def getGold(partition: str) -> list[list[tuple[str, list[tuple[int, int, str]], list[tuple[int, int, str]]]]]:
    return pickle.load(open(partition + "/" + partition.upper(), "rb"))

def detectRelation(first: Span, second: Span, sentence: Span):
    pattern = extractPattern(first, second, sentence)
    for label in labels:
        if pattern in goldPatterns[label]:
            return label
    return None

def extractPattern(first: Span, second: Span, sentence: Span) -> str:
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
    return " ".join(pattern)

