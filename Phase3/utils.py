import pickle
import spacy
from spacy.language import Language
from spacy.tokens.span import Span
from numpy import ndarray, array, concatenate
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

#models
ner: Language = pickle.load(open("NER", "rb")) 
nlp = ner #call it nlp when I am using it just for syntax
#classifier = pickle.load(open("classifier3", "rb"))

#vectors
dependencyVectors = pickle.load(open("Phase3/vectors/dependencyVectors", "rb"))



def extractRelations(docText: str) -> list[tuple[str, str, str, int]]:#TODO fix this to work like extractRelationsFromGoldEntities
    #the document is passed into the spacy model to extract drug entities
    doc = ner(docText)
    #extract all possible relations
    pairBins: HashBin[MentionPair, ndarray] = HashBin()
    for sentence in doc.sents:#the document is split into sentences
        ents = sentence.ents#spacy extracts the entities
        for i, first in enumerate(ents):
            for second in ents[i+1:]:
                if first.text.lower() != second.text.lower():
                    relation = MentionPair(first, second)
                    vector = pattern(first, second)
                    pairBins[relation] = vector#add vector to bin

    relations = list()#a container to hold extracted relations
    for pair, vectors in pairBins:#new part
        label = classify(vectors)
        if label is not None:
            relations.append((pair, label))
    return relations



def extractRelationsFromGoldEntities(doc: list[tuple[str, list[tuple[int, int, str]]]]) -> list[tuple[str, str, str, int]]:
    pairBins: HashBin[MentionPair, ndarray] = HashBin()
    for sentence in doc:#the document is split into sentences
        (sentenceText, drugs) = sentence
        ents: list[Span] = list()#container for ents
        sentence = nlp(sentenceText)
        for start, end, _ in drugs:#collect all the ents as Spans
            drug = sentence.char_span(start, end)
            if drug is not None:
                ents.append(drug)

        for i, first in enumerate(ents):
            for second in ents[i+1:]:
                for sent in sentence.sents:
                    if first.sent == sent and second.sent == sent:
                        if first.text.lower() != second.text.lower():
                            relation = MentionPair(first, second)
                            vector = pattern(first, second)
                            pairBins[relation] = vector#add vector to bin
        
    relations: list[tuple[str, str, str, int]] = list()#a container to hold extracted relations
    for pair, vectors in pairBins:
        label = classify(vectors)
        if label is not None:
            relations.append(pair)#TODO, make this actually "compile" with correct arguments
    return relations




def pattern(first: Span, second: Span):
    display = False
    if MentionPair(first.text,second.text) == MentionPair("ACEI", "ARB"):
        print([(w, w.i, w.head.i) for w in first.sent])
        display = True
    #I'm trying to standardize the vectors for paths between entities, so I'm going to put the earlier one alphabetically first, so that all paths between pairs of the same entities can be meaningfully compared
    swapped = 0
    if first.text.lower() > second.text.lower():
        temp = first; first = second; second = temp
        swapped = 1
    #find peak
    peak = None
    pointer1 = first.root; path1len = 0#initialize left side of path
    while peak is None: #wait to find a "lowest common dependency ancestor"
        pointer2 = second.root; path2len = 0#initialize right side of path
        while peak is None:
            if display:
                print((pointer1, pointer1.dep_), (pointer2, pointer2.dep_))
                input()
            if pointer1 == pointer2:#when they line up
                peak = pointer1#that's the peak
            if pointer2.head == pointer2:
                break
            pointer2 = pointer2.head; path2len += 1 #otherwise, iterate right
        if pointer1.head == pointer1:
            break
        pointer1 = pointer1.head; path1len += 1 #iterate left
    if peak is None:
        raise Exception("No dependency peak found!")
    print("found peak")
    #calculate first side
    path1 = array([0.0 for _ in range(300)])
    dep1 =  array([0.0 for _ in range(600)])
    pointer = first.root; height = 0
    while pointer != peak and pointer.head != peak:
        """if height > path1len:
            print(first.root.i, second.root.i)
            print([(w, w.i, w.head.i) for w in pointer.sent])
            input()#using this to debug"""
        path1 += height/path1len * pointer.vector
        pointer = pointer.head; height += 1
        dep1 += height/path1len * dependencyVectors[pointer.dep_]
    #calculate second side
    path2 = array([0 for _ in range(300)])
    dep2 =  array([0 for _ in range(600)])
    pointer = second.root; height = 0
    while pointer != peak and pointer.head != peak:
        """if height > path2len:
            print(first.root.i, second.root.i)
            print([(w, w.i, w.head.i) for w in pointer.sent])
            input()#using this to debug"""
        path1 += height/path2len * pointer.vector
        pointer = pointer.head; height += 1
        dep1 += height/path2len * dependencyVectors[pointer.dep_]

    return concatenate(([path1len], path1, dep1, peak.vector, dep2, path2, [path2len], [swapped]))





def classify(vectors: set[DataFrame]):
    predictions = [classifier.predict(vector)[0] for vector in vectors]#get all predictions
    predictions = [(predictions.count(prediction), prediction) for prediction in set(predictions)]#find how common each is
    predictions.sort(key = lambda x : -x[0])#sort
    predictions = [prediction for prediction in predictions if prediction[0] == predictions[0][0]]#keep only the most common predictions
    if predictions[0] == "none":
        del predictions[0]
    if len(predictions) == 1 or len(predictions) == 2 and predictions[1] == "none":#if there is only a single most predicted prediction that isn't "none"
        return predictions[0]#return it











#This is a helper class I wrote to simplify case-insensitive mention pair encapsulation. It's basically just a wrapper class for a two-element  tuple that ignores case and order when compared
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
        return str(self())
    def __repr__(self):
        return repr(self())
    def __getitem__(self, key):
        return self().__getitem__(key)
    def __iter__(self):
        return self().__iter__()
    def __call__(self):
        return (self.one, self.two)

#this is a helper class I wrote to simplify the process of collecting MentionPairs and associating them with sets of DataFrame vectors. It will be used as basically just a dictionary from a tuple of Spans to a set of DataFrames 
from typing import TypeVar, Generic, Mapping
T = TypeVar("T")
E = TypeVar("E")
class HashBin(Mapping[T, set[E]]):
    def __init__(self):
        self.data: dict[T, set[E]] = dict()
    def __setitem__(self, key: T, newvalue: E) -> None:#this method is kinda confusing, basically if bin is a HashBin, the line bin[1] = 'a' adds 'a' to the 1 bin, not overwrites it. so I could then write bin[1] = 'b', then call bin[1], which would return {'a','b'}. I couldn't figure out how to make it use += instead
        bin: set[E] = self.data.get(key)
        if bin is None:
            bin = set()
            self.data[key] = bin
        bin.add(newvalue)
    def __getitem__(self, key: T) -> set[E]:
        bin = self.data.get(key)
        if bin is None:
            return {}
        return bin
    def __eq__(self, other) -> bool:
        if type(other) is not HashBin:
            return False
        return self.data == other.data
    def __ne__(self, other) -> bool:
        return not self == other
    def __str__(self) -> str:
        return self.data.__str__()
    def __repr__(self) -> str:
        return self.data.__repr__()
    def __iter__(self):
        return HashBin.Iter(self)
    class Iter(Generic[T]):
        def __init__(self, hashBin):
            self.data = hashBin.data
            self.iter = self.data.__iter__()
        def __next__(self) -> tuple[T, set[E]]:
            key = self.iter.__next__()
            return (key, self.data[key])