import os
from xml.etree import ElementTree as ET
import utils
from spacy import displacy

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


for i in range(3):
    displacy.serve(utils.ner(sentenceList[i]), style = "dep")