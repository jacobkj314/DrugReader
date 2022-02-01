import os
from xml.etree import ElementTree as ET
import spacy
import pickle

def getDEV() -> list[set[tuple[int, int, str]]]:
    return pickle.load(open("Dev/DEV", "rb"))
dev = getDEV()

def getNER() -> spacy.language.Language:
    return pickle.load(open("NER", "rb")) 
ner = getNER()
def extractEntities(sentenceText: str) -> set[tuple[int, int, str]]:
    entities = set()
    sentence = ner(sentenceText)#convert text to spacy model
    for ent in sentence.ents:
        entities.add((ent.start_char, ent.end_char, ent.label_))
    return entities



sentenceList = list()
docFolder = "Dev"
for file in os.listdir(docFolder): #get every training document
    if str(file) == "DEV":
        continue
    file = os.path.join(docFolder, file)
    #doc = open(file, "r").read()
    root = ET.parse(file).getroot() #parse document as XML tree
    for sentence in root.iter("sentence"):
        sentenceText = sentence.get("text")
        sentenceList.append(sentenceText)



extractedList = list()
answers, guesses, hits = 0,0,0
i = 0
while i < len(dev) and i < len(sentenceList):
    #gold = dev[i]
    gold = {t[:2] for t in dev[i]}
    #extractedEntities = extractEntities(sentenceList[i])
    extractedEntities = {t[:2] for t in extractEntities(sentenceList[i])}
    answers += len(gold)
    guesses += len(extractedEntities)
    hits += len(gold.intersection(extractedEntities))
    #precision = 0 if (guesses == 0) else hits/guesses
    #recall = 1 if (answers == 0) else hits/answers
    #print("Answers:" + str(answers) + " Guesses: " + str(guesses) + " Precision:" + str(precision) + " Recall:" + str(recall))
    #print(str(gold) + "\n" + str(extractedEntities) + "\n")
    i += 1

precision, recall = hits/guesses, hits/answers
fscore = (2 * precision * recall)/(precision + recall)

print("Precision:" + str(hits/guesses) + " Recall:" + str(hits/answers))
print("F-score:" + str(fscore))







