import pickle
import spacy
from spacy.language import Language
from spacy.tokens.span import Span
from numpy import ndarray, array
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression

#architecture parameters:
maskEntitiesInPath = False
resolveCollisions = True

#language models
ner: Language = pickle.load(open("../NER", "rb")) 
nlp = ner #call it nlp when I am using it just for syntax

#multiclassifier
multiclassifier: LogisticRegression = pickle.load(open("models/multiclassifier", "rb"))

#multibinary classifiers
mechanismClassifier: LogisticRegression = pickle.load(open("models/mechanismClassifier", "rb"))
effectClassifier: LogisticRegression = pickle.load(open("models/effectClassifier", "rb"))
adviseClassifier: LogisticRegression = pickle.load(open("models/adviseClassifier", "rb"))
intClassifier: LogisticRegression = pickle.load(open("models/intClassifier", "rb"))
def multibinaryClassify(vector:ndarray) -> set[str]:#helper method to simplify classification
    result = set()
    if mechanismClassifier.predict(vector)[0] == "true":
        result.add("mechanism")
    if effectClassifier.predict(vector)[0] == "true":
        result.add("effect")
    if adviseClassifier.predict(vector)[0] == "true":
        result.add("advise")
    if intClassifier.predict(vector)[0] == "true":
        result.add("int")
    return result

#pipeline classifiers
pipelineMain: LogisticRegression = pickle.load(open("models/pipeline-main", "rb"))
pipelineMulti: LogisticRegression = pickle.load(open("models/pipeline-multi", "rb"))

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
        (sentenceText, drugs, interactions) = sentence
        ents: list[Span] = list()#container for ents
        sentence = nlp(sentenceText)
        for start, end, _ in drugs:#collect all the ents as Spans
            drug = sentence.char_span(start, end)
            if drug is not None:
                ents.append(drug)

        entsStr = [ent.text for ent in ents]

        entCount = len(ents)
        #for i, first in enumerate(ents):
        #    for second in [ents[j] for j in range(i + 1, entCount, 1)]:
        for first, second, _ in interactions:
            first = ents[first]
            second = ents[second]
            for _ in range(1):
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
    if maskEntitiesInPath:
        path1 = [point for point in path1 if point not in ents]
        path2 = [point for point in path2 if point not in ents]

    
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

def detectRelationMultiBinary(first: Span, second: Span, sentence: Span, ents: list[str]):
    pattern = extractPattern(first, second, sentence, ents)
    predictions = multibinaryClassify(pattern)
    if resolveCollisions:
        multiAnswer = pipelineMulti.predict(pattern)[0]
        if multiAnswer in predictions:
            return multiAnswer
    else:
        if len(predictions) == 1:#only select if there is a single one that looks right
            return predictions.pop()

def detectRelationPipeline(first: Span, second: Span, sentence: Span, ents: list[str]):
    pattern = extractPattern(first, second, sentence, ents)
    if pipelineMain.predict(pattern) == "true":
        return pipelineMulti.predict(pattern)[0]





def filter(relations):
    return [(first, second, label, sentence) for first, second, label, sentence in relations if first.lower() != second.lower()]



detectRelation = detectRelationMulticlass #set which detection scheme to use by default