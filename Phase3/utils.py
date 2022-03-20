import pickle
import spacy
from spacy.language import Language
from spacy.tokens.span import Span
from numpy import ndarray, array
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

#language models
ner: Language = pickle.load(open("NER", "rb")) 
nlp = ner #call it nlp when I am using it just for syntax

#multiclassifier
multiclassifier: LogisticRegression = pickle.load(open("Phase3/models/multiclassifier", "rb"))

#Helper method to load gold data
def getGold(partition: str) -> list[list[tuple[str, list[tuple[int, int, str]], list[tuple[int, int, str]]]]]:
    return pickle.load(open("../" + partition + "/" + partition.upper(), "rb"))

def extractRelations(docText: str) -> list[tuple[str, str, str, int]]:
    #the document is passed into the spacy model to extract drug entities
    doc = ner(docText)
    #extract all possible relations
    relations = list()#a container to hold extracted relations
    for s, sentence in enumerate(doc.sents):#the document is split into sentences
        ents = sentence.ents#spacy extracts the entities
        entsStr = [ent.text for ent in ents]
        entCount = len(ents)
        for i, first in enumerate(ents):
            for second in [ents[j] for j in range(i + 1, entCount, 1)]:
                relation = detectRelation(first, second, sentence, entsStr)#each potential pair is compared
                if relation is not None:
                    relations.append((first.text, second.text, relation, s))#if there is a relation, add it to the extracted relation

    relations = filter(relations)
    return relations

def extractRelationsFromGoldEntities(doc: list[tuple[str, list[tuple[int, int, str]]]]) -> list[tuple[str, str, str, int]]:
    relations: list[tuple[str, str, str, int]] = list()#a container to hold extracted relations
    for s, sentence in enumerate(doc):#the document is split into sentences
        (sentenceText, drugs) = sentence
        ents: list[Span] = list()#container for ents
        sentence = nlp(sentenceText)
        for start, end, _ in drugs:#collect all the ents as Spans
            drug = sentence.char_span(start, end)
            if drug is not None:
                ents.append(drug)

        entsStr = [ent.text for ent in ents]

        entCount = len(ents)
        for i, first in enumerate(ents):
            for second in [ents[j] for j in range(i + 1, entCount, 1)]:
                for sent in sentence.sents:
                    if first.sent == sent and second.sent == sent:
                        relation = detectRelation(first, second, sent, entsStr)#each potential pair is compared
                        if relation is not None:
                            relations.append((first.text, second.text, relation, s))#if there is a relation, add it to the extracted relations
                        break

    relations = filter(relations)
    return relations



def extractPattern(first: Span, second: Span, sentence: Span, ents: list[str]) -> DataFrame:
    #accumulator lists
    path1: list[str] = list()
    path2: list[str] = list()
    #sentence root
    root = sentence.root
    #tracing the entities up the dependency tree
    pointer1 = first.root#first
    path1.append(pointer1.lemma_)
    while pointer1 != root:
        path1.append(pointer1.head.lemma_)
        pointer1 = pointer1.head
    pointer2 = second.root#second
    path2.append(pointer2.lemma_)
    while pointer2 != root:
        path2.append(pointer2.head.lemma_)
        pointer2 = pointer2.head
    #remove duplicates
    i = -1
    while len(path1) >= abs(i) and len(path2) >= abs(i) and path1[i] == path2[i]:
        i -= 1
    i += 1 #this is the index of the first node the paths have in common

    peak = path1[i]#"common ancestor" of entities in dependency parse
    path1 = path1[1:i]#path leading from entity 1 to peak
    path2 = path2[1:i]#path leading from entity 2 to peak

    
    path1len = len(path1)
    path2len = len(path2)
    #add up the vectors, with the vectors coming further away having more of an influence on the final pattern
    pattern = nlp(peak).vector
    for j in range(path1len):
        pattern += (j+1)/(path1len + 1) * nlp(path1[j]).vector
    for j in range(path2len):
        pattern += (j+1)/(path2len + 1) * nlp(path2[j]).vector

    pattern = array([pattern])#rotate to row vector
    pattern = DataFrame(pattern)#convert to pandas dataframe
    return pattern

def detectRelationMulticlass(first: Span, second: Span, sentence: Span, ents: list[str]):
    pattern = extractPattern(first, second, sentence, ents)
    label = multiclassifier.predict(pattern)[0]
    if label != "none":
        return label

def filter(relations):
    return [(first, second, label, sentence) for first, second, label, sentence in relations if first.lower() != second.lower()]

#This is a helper class I wrote to simplify case-insensitive mention pair encapsulation. It's basically just a wrapper class for a tuple that ignores case and order when compared
class MentionPair:
    def __init__(self, one, two):
        self.one = one
        self.two = two
    def __eq__(self, other):
        if type(other) is not MentionPair:
            return False
        return {str(self.one).lower(), str(self.two).lower()} == {str(other.one).lower(), str(other.two).lower()}
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash(tuple(sorted([str(self.one).lower(), str(self.two).lower()])))
    def __str__(self):
        return str(self.get())
    def __repr__(self):
        return repr(self.get())
    def __getitem__(self, key):
        return self.get().__getitem__(key)
    def __iter__(self):
        return self.get().__iter__()
    def get(self):
        return (self.one, self.two)

#this is a helper class I wrote to simplify the process of collecting MentionPairs and associating them with sets of DataFrame vectors
from typing import TypeVar, Generic
T = TypeVar("T")
class HashBin(Generic[T]):
    def __init__(self):
        self.data: dict[T, set[T]] = dict()
    def __setitem__(self, key: T, newvalue):#this method is kinda confusing, basically if bin is a HashBin, the line bin[1] = 'a' adds 'a' to the 1 bin, not overwrites it. so I could then write bin[1] = 'b', then call bin[1], which would return {'a','b'}. I couldn't figure out how to make it use += instead
        bin = self.data.get(key)
        if bin is None:
            bin = set()
            self.data[key] = bin
        bin.add(newvalue)
    def __getitem__(self, key: T):
        bin = self.data.get(key)
        if bin is None:
            return {}
        return bin
    def __iter__(self):
        return HashBin.Iter(self)
    class Iter(Generic[T]):
        def __init__(self, hashBin):
            self.data = hashBin.data
            self.iter = self.data.__iter__()
        def __next__(self):
            key = self.iter.__next__()
            return (key, self.data[key])