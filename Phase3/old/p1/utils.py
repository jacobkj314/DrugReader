import pickle
import spacy
from spacy.language import Language
from spacy.tokens.span import Span
from spacy.tokens.doc import Doc
from numpy import ndarray, std
from scipy import spatial
from statistics import mean

#architecture parameters
thr: float = .55
vectorType = "peak-weighted"
removedByAnova = 0

#language models
ner: Language = pickle.load(open("NER", "rb")) 
nlp = ner #call it nlp when I am using it just for syntax

labels = ["effect", "mechanism", "advise", "int"]
#endVectors = pickle.load(open("goldVectors_3-4-end", "rb"))
peakVectors = None# #pickle.load(open("goldVectors_3-4-peak", "rb"))#only loading one set of vectors to save on memory
#uniformVectors = pickle.load(open("goldVectors0_1-2", "rb"))
def goldVectors() -> dict[str, list[ndarray]]:
    """if vectorType == "end-weighted":
        return endVectors
    elif vectorType == "peak-weighted":
        return peakVectors
    elif vectorType == "uniform-weighted":
        return uniformVectors
    else:
        raise ValueError("invalid vectorType")"""
    return peakVectors

#Helper method to load gold data
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
                relation = detectRelationAnova(first, second, sentence)#each potential pair is compared
                if relation is not None:
                    relations.append((first.text, second.text, relation, s))#if there is a relation, add it to the extracted relations
    return relations

def extractRelationsFromGoldEntities(doc: list[tuple[str, list[tuple[int, int, str]]]]) -> list[tuple[str, str, str, int]]:
    relations = list()#a container to hold extracted relations
    for s, sentence in enumerate(doc):#the document is split into sentences
        (sentenceText, drugs) = sentence
        ents: list[Span] = list()#container for ents
        sentence = nlp(sentenceText)
        for start, end, _ in drugs:#collect all the ents as Spans
            drug = sentence.char_span(start, end)
            if drug is not None:
                ents.append(drug)

        entCount = len(ents)
        for i, first in enumerate(ents):
            for second in [ents[j] for j in range(i + 1, entCount, 1)]:
                for sent in sentence.sents:
                    if first.sent == sent and second.sent == sent:
                        relation = detectRelationAnova(first, second, sent)#each potential pair is compared
                        if relation is not None:
                            relations.append((first.text, second.text, relation, s))#if there is a relation, add it to the extracted relations
    return relations

def cosine(v: ndarray, u: ndarray) -> float:
    return 1 - spatial.distance.cosine(v, u)

def meanSdCount(values: list[float]) -> tuple[float, float, int]:#output tuples of the form mean, sd, count for the group
    return mean(values), std(values), len(values)

def anova(groups: list[tuple[float, float, int]]) -> float: #input tuples of the form mean, sd, count for each group
    meanOfMeans = mean([group[0] for group in groups])
    sumSquaresBetween = sum([group[2] * (meanOfMeans - group[0])**2 for group in groups])
    sumSquaresWithin = sum([(group[1]**2) * (group[2] - 1) for group in groups])
    dfBetween = len(groups) - 1
    dfWithin = sum([group[2] for group in groups]) - len(groups)
    return (sumSquaresBetween/dfBetween)/(sumSquaresWithin/dfWithin)

#compares the extracted context vector of the two entities to known gold entities and returns a chosen label
def detectRelationAnova(first: Span, second: Span, sentence: Span):
    global removedByAnova
    vector = extractPattern(first, second, sentence)
    sims = [meanSdCount([cosine(vector, v) for v in goldVectors()[label]]) for label in labels]
    fWith = anova(sims)
    means = [sim[0] for sim in sims]
    maxMean = max(means)
    if maxMean > thr:
        for i, _ in enumerate(sims):
            if sims[i][0] == maxMean:
                del sims[i]
                fWithout = anova(sims)
                if fWithout < fWith:
                    return labels[i]
                else:
                    removedByAnova += 1
                    break
    return None

#Extracts the syntactic pattern joining two entities as a vector
def extractPattern(first: Span, second: Span, sentence: Span) -> ndarray:
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

    pattern = None
    if vectorType == "peak-weighted":
        path1len = len(path1)
        path2len = len(path2)

        #add up the vectors, with the vectors coming further away having more of an influence on the final pattern
        pattern = nlp(peak).vector
        for j in range(path1len):
            pattern += (j+1)/(path1len + 1) * nlp(path1[j]).vector
        for j in range(path2len):
            pattern += (j+1)/(path2len + 1) * nlp(path2[j]).vector
    elif vectorType == "end-weighted":
        path1 = path1[::-1]
        path2 = path2[::-1]

        path1len = len(path1)
        path2len = len(path2)

        #add up the vectors, with the vectors coming further away having less of an influence on the final pattern
        pattern = nlp(peak).vector / ((path1len if path1len != 0 else 1) * (path2len if path2len != 0 else 1))
        for j in range(path1len):
            pattern += (j+1)/(path1len + 1) * nlp(path1[j]).vector
        for j in range(path2len):
            pattern += (j+1)/(path2len + 1) * nlp(path2[j]).vector
    elif vectorType == "uniform-weighted":
        pattern = nlp(peak).vector
        for j in path1:
            pattern += nlp(j).vector
        for j in path2:
            pattern += nlp(j).vector
    else:
        raise ValueError("invalid vectorType")

    return pattern

