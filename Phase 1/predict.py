import os
from xml.etree import ElementTree as ET
import spacy
import pickle

def getDEV() -> list[set[tuple[int, int, str]]]:
    return pickle.load("Dev/DEV")
dev = getDEV()

def getNER() -> spacy.language.Language:
    return pickle.load("NER") 
ner = getNER()
def extractEntities(sentenceText: str) -> set[tuple[int, int, str]]:
    entities = set()
    sentence = ner(sentenceText)#convert text to spacy model
    for ent in sentence.ents:
        entities.add((ent.start, ent.end, ent.label_))
    return entities



sentenceList = list()
docFolder = "Train"
for file in os.listdir(docFolder): #get every training document
    file = os.path.join(docFolder, file)
    #doc = open(file, "r").read()
    root = ET.parse(file).getroot() #parse document as XML tree
    for sentence in root.iter("sentence"):
        sentenceText = sentence.get("text")
        sentenceList.append(sentenceText)

extractedList = list()
i = 0
while i < len(dev) and i < len(sentenceList):
    gold = dev[i]
    extractedEntities = extractEntities(sentenceList[i])
    answers = len(gold)
    guesses = len(extractedEntities)
    hits = len(gold.intersection(extractedEntities))
    print("Precision:" + (hits/guesses) + " Recall:" + (hits/answers))








