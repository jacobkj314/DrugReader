import pickle
import spacy
from spacy.language import Language
from spacy.tokens.span import Span
from spacy.tokens.doc import Doc
from scipy import stats, spatial
from numpy import ndarray
from statistics import mean

ner: Language = pickle.load(open("NER", "rb")) 
nlp = spacy.load("en_core_web_lg")

labels = ["effect", "mechanism", "advise", "int"]
goldVectors: dict[str, list[ndarray]] = pickle.load(open("goldVectors", "rb"))

thr: float = 0

def getGold(partition: str) -> list[list[tuple[str, list[tuple[int, int, str]], list[tuple[int, int, str]]]]]:
    return pickle.load(open(partition + "/" + partition.upper(), "rb"))

def extractRelations(docText: str) -> list[tuple[str, str, str, int]]:
    #the document is passed into the spacy model to extract drug entities
    doc = ner(docText)
    #extract all possible relations
    relations = list()#a container to hold extracted relations
    for s, sentence in enumerate(doc.sents):#the document is split into sentences
        ents = sentence.ents#spacy extracts the entities
        entCount = len(ents)
        for i, first in enumerate(ents):
            for second in [ents[j] for j in range(i + 1, entCount, 1)]:
                relation = detectRelation1(first, second, sentence)#each potential pair is compared
                if relation is not None:
                    relations.append((first.text, second.text, relation, s))#if there is a relation, add it to the extracted relations
    return relations


def cosine(v: ndarray, u: ndarray) -> float:
    return 1 - spatial.distance.cosine(v, u)


def detectRelation1(first: Span, second: Span, sentence: Span):
    vector = nlp(extractPattern(first, second, sentence)).vector
    sims = [[cosine(vector, v) for v in goldVectors[label]] for label in labels]
    _, pWith = stats.f_oneway(sims[0], sims[1], sims[2], sims[3])
    means = [mean(sim) for sim in sims]
    maxMean = max(means)
    if maxMean > thr:
        for i, _ in enumerate(sims):
            if mean(sims[i]) == maxMean:
                del sims[i]
                _, pWithout = stats.f_oneway(sims[0], sims[1], sims[2])
                if pWithout > pWith:
                    return labels[i]
                else:
                    break
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

