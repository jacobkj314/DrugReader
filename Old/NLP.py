from io import TextIOWrapper
import sys
from typing import Text
import spacy
from spacy.tokens import doc
from spacy.tokens.span import Span
import pickle
#from sklearn.feature_extraction import DictVectorizer
#import pandas as pd
import IEutils


def dlramt(storyNLP : doc.Doc) -> set[Span]:
    dlramts = set()

    for i, token in enumerate(storyNLP):
        if token.text.lower() == "disclosed":
            if token.i > 0 and token.nbor(-1).text == "not":
                dlramts.add(storyNLP[i-1:i+1])
                return dlramts
        elif token.text.lower() == "undisclosed":
            if token.i < len(token.doc) -1 and token.nbor(1).text == "amount":
                dlramts.add(storyNLP[i:i+2])
                return dlramts
            dlramts.add(storyNLP[i:i+1])
            return dlramts
        elif token.is_currency:
            if token.head.lemma_ in ["for", "pay"]: 
                dlramts.add(storyNLP[token.left_edge.i:token.right_edge.i])
    
    for ent in storyNLP.ents:
        if ent.label_ == "MONEY":
            if ent.root.head.lemma_ in ["for", "pay"]: 
                dlramts.add(ent)

    return dlramts

def acqloc(storyNLP : doc.Doc) -> set[Span]:
    potentialLocs = list[Span]()
    for ent in storyNLP.ents:
        if ent.label_ == "GPE": 
            potentialLocs.append(ent)
    
    potentialLocsCombined = list[Span]()
    i = 0
    while i < len(potentialLocs):
        ent = potentialLocs[i]
        if i + 1 < len(potentialLocs):
            e = potentialLocs[i + 1]
            if e.start - ent.end == 1 and storyNLP[ent.end].text == ",": #if the entities are separated only by a comma
                potentialLocsCombined.append(storyNLP[ent.start:e.end])
                i+=1
            else:
                potentialLocsCombined.append(ent)
        else:
            potentialLocsCombined.append(ent)
        
        i+=1
        
    acqlocs = set()
    for ent in potentialLocsCombined:
        if ent.root.head.lemma_ in ["in", "of", "from", "at"] : 
            acqlocs.add(ent)
        if ent.root.head.ent_type_ == "ORG" and ent.start < ent.root.head.i:
            acqlocs.add(ent) 

    return acqlocs


def getAcqbusesStatuses(storyNLP : doc.Doc, v, bigBrain) -> tuple[set[Span], set[Span]] :
    
    acqbuses = set()
    statuses = set()
    
    #We tried to use sklearn to use ML to predict these, but it turns out spaCy does a better job
    """ testData = IEutils.convertDocToDict(storyNLP)
    
    testDataToDict = testData.to_dict('records')
    transformedData = v.transform(testDataToDict)
    labels = bigBrain.predict(transformedData)
    
    i = 0
    while i < len(labels):
        if(labels[i] == "ACQBUS"):
            start = i
            while i < len(labels) and (labels[i] == "ACQBUS"):
                i += 1
            acqbuses.add(storyNLP[start:i])    
        elif(labels[i] == "STATUS"):
            start = i
            while i < len(labels) and (labels[i] == "STATUS"):
                i += 1        
            statuses.add(storyNLP[start:i])
        i += 1 """

    return acqbuses, statuses


def getAcquiredPurchaserSeller(storyNLP : doc.Doc) -> tuple[set[Span], set[Span], set[Span]]:
    #place to hold what we find
    acquireds = set()
    purchasers = set()
    sellers = set()

    for sentenceIndex, sentence in enumerate(storyNLP.sents):
        verb = IEutils.getVerb(sentence)
        #it looks like we don't need to worry about complementizers like "that" 
        #but this also ignores modal verbs that we might want to consider

        if verb.lemma_ in ["buy", "purchase", "acquire", "obtain", "get", "secure", "procure", "gain"]:
            for entity in (entity for entity in sentence.ents if entity.label_ in ["ORG", "PERSON"]):
                if entity.root.dep_ == "nsubj":
                    purchasers.add(entity)
                elif entity.root.dep_ == "nsubjpass":
                    acquireds.add(entity)
                elif entity.root.dep_ == "dobj":
                    acquireds.add(entity)
                elif entity.root.head.lemma_ == "by":
                    purchasers.add(entity)
                elif entity.root.head.lemma_ == "from":
                    sellers.add(entity)
        elif verb.lemma_ in ["sell", "exchange"]:
            for entity in (entity for entity in sentence.ents if entity.label_ in ["ORG", "PERSON"]):
                if entity.root.dep_ == "nsubj":
                    sellers.add(entity)
                elif entity.root.dep_ == "nsubjpass":
                    acquireds.add(entity)
                elif entity.root.dep_ == "dobj":
                    acquireds.add(entity)
                elif entity.root.head.lemma_ == "by":
                    sellers.add(entity)
                elif entity.root.head.lemma_ == "to":
                    purchasers.add(entity)

    #We tried to make more rules, but they didn't improve our scores
    """for token in storyNLP:
        if token.lemma_ in ["buy", "purchase", "acquire", "obtain", "get", "secure", "procure", "gain"]:
            for child in token.children:
                if child.dep_ == "nsubj":
                    purchasers.add(storyNLP[child.left_edge.i:child.right_edge.i])
                elif child.dep_ == "nsubjpass":
                    acquireds.add(storyNLP[child.left_edge.i:child.right_edge.i])
                elif child.dep_ == "dobj":
                    acquireds.add(storyNLP[child.left_edge.i:child.right_edge.i])
                elif child.lemma_ == "by":
                    purchasers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])
                elif child.lemma_ == "from":
                    sellers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])
        elif token.lemma_ in ["sell", "exchange"]:
            for child in token.children:
                if child.dep_ == "nsubj":
                    sellers.add(storyNLP[child.left_edge.i:child.right_edge.i])
                elif child.dep_ == "nsubjpass":
                    acquireds.add(storyNLP[child.left_edge.i:child.right_edge.i])
                elif child.dep_ == "dobj":
                    acquireds.add(storyNLP[child.left_edge.i:child.right_edge.i])
                elif child.lemma_ == "by":
                    sellers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])
                elif child.lemma_ == "to":
                    purchasers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])"""



    for entity in (entity for entity in storyNLP.ents if entity.label_ in ["ORG", "PERSON"]):
        if entity not in acquireds and entity not in purchasers and entity not in sellers:
            if entity.root.head.text.lower() == "of" and entity.root.head.head.text.lower() in ["acquisition", "purchase"]:
                acquireds.add(entity)
                for child in entity.root.head.head.children:
                    if child.text.lower() == "by":#acquisition of ___ by ...
                        purchasers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])
                    else:
                        for ent in entity.sent.ents:
                            if ent.end < entity.start:#this comes before the main entity
                                purchasers.add(ent)
                                break
                    if child.text.lower() == "from": #purchase of ___ from ...
                        sellers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])
            elif entity.root.head.text.lower() == "of" and entity.root.head.head.text.lower() in ["sale"]:
                acquireds.add(entity)
                for child in entity.root.head.head.children:
                    if child.text.lower() == "by":#sale of ___ by ...
                        sellers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])
                    else:
                        for ent in entity.sent.ents:
                            if ent.end < entity.start:#this comes before the main entity
                                sellers.add(ent)
                                break
                    if child.text.lower() == "to": #sale of ___ to ...
                        purchasers.add(storyNLP[child.left_edge.i+1:child.right_edge.i])


    return acquireds, purchasers, sellers

def getAll(story: doc.Doc):
    acquireds = set()
    acqbuses = set()
    acqlocs = set()
    dlramts = set()
    purchasers = set()
    sellers = set()
    statuses = set()

    for ent in story.ents : 
        if ent.label_ == "ACQUIRED" : 
            acquireds.add(ent)
        elif ent.label_ == "ACQBUS" : 
            acqbuses.add(ent)
        elif ent.label_ == "ACQLOC" : 
            acqlocs.add(ent)
        elif ent.label_ == "DLRAMT" : 
            dlramts.add(ent)
        elif ent.label_ == "PURCHASER" : 
            purchasers.add(ent)
        elif ent.label_ == "SELLER" : 
            sellers.add(ent)
        elif ent.label_ == "STATUS" : 
            statuses.add(ent)

    return acquireds, acqbuses, acqlocs, dlramts, purchasers, sellers, statuses



def write(writer : TextIOWrapper, entsOG : set[Span], label: str) -> None:
    ents = list(set([ent.text for ent in entsOG if ent.text != ""]))
    if len(ents) == 0: #if its empty
        writer.write(f"{label}: ---\n")
        return
    
    #this loop prevents duplicate outputs like "Salt Lake City, UT" and "Salt Lake City", always preferring the longer one
    entsFinal = list()
    i = 0
    while i < len(ents):
        needToAddI = True
        j = 0
        while j < len(ents):
            if i != j:
                if ents[i] in ents[j]:#if they are containing each other
                    if len(ents[i]) < len(ents[j]):
                        needToAddI = False
                        break
            j += 1
        if needToAddI:
            entsFinal.append(ents[i])
        i += 1

    for ent in entsFinal:
        writer.write(f"{label}: \"{ent}\"\n")



def main():
    nlp = spacy.load("en_core_web_lg") #!python -m spacy download en_core_web_lg
    ner = pickle.load(open("NER", "rb"))

    storyDict = dict()#store stories
    storyIDs = list()#store the order of stories
    def stories() -> tuple[str, doc.Doc] :
        for storyID in storyIDs:
            yield storyID, storyDict[storyID]

    #Getting the training data - not using sklearn anymore
    #(v, bigBrain) = pickle.load(open("trainData", "rb"))

    #read from doclist
    with open(sys.argv[1], "r") as docListReader:
	    docList = docListReader.read()
    #iterate over the files
    for file in docList.splitlines():
        fileName = file.split("/")[-1]
        storyIDs.append(fileName)#add ID
        with open(file, "r") as docReader:
            text = docReader.read().replace("\n", " ") #delete newlines and create story object
            storyNLP = nlp(text)
            storyNER = ner(text)
            storyDict[fileName] = (storyNLP, storyNER) #add story as tupe of nlp and ner

    with open(sys.argv[1] + ".templates", "w") as writer:
        for ID, story in stories(): #story is a tuple of the nlp and ner
            storyNLP, storyNER = story[0], story[1]

            #Get predictions from fine-tuned NER model
            acquireds, acqbuses, acqlocs, dlramts, purchasers, sellers, statuses = getAll(storyNER)

            #We tried to train up 3 models and only select the responses that two or more agreed on, but this didn't help our score
            """acquireds2, acqbuses2, acqlocs2, dlramts2, purchasers2, sellers2, statuses2 = getAll(storyNER2)
            acquireds3, acqbuses3, acqlocs3, dlramts3, purchasers3, sellers3, statuses3 = getAll(storyNER3)

            acquireds = (acquireds.intersection(acquireds2)).union(acquireds2.intersection(acquireds3)).union(acquireds.intersection(acquireds3))
            acqbuses = (acqbuses.intersection(acqbuses2)).union(acqbuses2.intersection(acqbuses3)).union(acqbuses.intersection(acqbuses3))
            acqlocs = (acqlocs.intersection(acqlocs2)).union(acqlocs2.intersection(acqlocs3)).union(acqlocs.intersection(acqlocs3))
            dlramts = (dlramts.intersection(dlramts2)).union(dlramts2.intersection(dlramts3)).union(dlramts.intersection(dlramts3))
            purchasers = (purchasers.intersection(purchasers2)).union(purchasers2.intersection(purchasers3)).union(purchasers.intersection(purchasers3))
            sellers = (sellers.intersection(sellers2)).union(sellers2.intersection(sellers3)).union(sellers.intersection(sellers3))
            statuses = (statuses.intersection(statuses2)).union(statuses2.intersection(statuses3)).union(statuses.intersection(statuses3))"""

            #Get predictions from our rules
            acquireds2, purchasers2, sellers2 = getAcquiredPurchaserSeller(storyNLP)
            #acqbuses2, statuses2 = getAcqbusesStatuses(storyNLP, v, bigBrain)
            acqlocs2 = acqloc(storyNLP)
            dlramts2 = dlramt(storyNLP)

            #combine them
            acquireds = acquireds.union(acquireds2)
            #acqbuses = acqbuses.union(acqbuses2)
            acqlocs = acqlocs.union(acqlocs2)
            dlramts = dlramts.union(dlramts2)
            purchasers = purchasers.union(purchasers2)
            sellers = sellers.union(sellers2)
            #statuses = statuses.union(statuses2)

            #filter out some extra guesses
            statuses = filterStatuses(statuses)
            acquireds = filterEntities(acquireds)
            purchasers = filterEntities(purchasers)
            sellers = filterEntities(sellers)

            #write everything to the output
            writer.write(f"TEXT: {ID}\n")
            write(writer, acquireds, "ACQUIRED")
            write(writer, acqbuses, "ACQBUS")
            write(writer, acqlocs, "ACQLOC")
            write(writer, dlramts, "DLRAMT")
            write(writer, purchasers, "PURCHASER")
            write(writer, sellers, "SELLER")
            write(writer, statuses, "STATUS")
            writer.write("\n")
        
        writer.close()


def filterStatuses(statuses):
    filteredStatuses = set()
    if len(statuses) > 0:
        statusList = list(statuses)
        statusList.sort( key = lambda x : len(x) )
        filteredStatuses.add(statusList[0])
    return filteredStatuses

def filterEntities(spans : set[Span]):
    filteredEntities = set()
    for span in spans:
        if span.root.pos_ in ["NOUN", "PROPN"]:
            filteredEntities.add(span)
    return filteredEntities

if __name__ == "__main__":
    main()