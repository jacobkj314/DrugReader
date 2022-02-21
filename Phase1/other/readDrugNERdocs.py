#this code reads the annotated documents and "pickles" them, so they can be accessed faster than reading them from xml
import os
from xml.etree import ElementTree as ET
import pickle

def readDrugNER(docFolder : str) -> list[set[tuple[int, int, str]]]:
    docList = list()

    for file in os.listdir(docFolder): #get every training document
        if str(file) == "TEST":#ignore pickled file
            continue
        file = os.path.join(docFolder, file)
        root = ET.parse(file).getroot() #parse document as XML tree
        sentenceList = list()
        for sentence in root.iter("sentence"):
            sentenceText = sentence.get("text")
            drugList = list()
            interactionList = list()
            for drug in sentence.iter("entity"):
                chars = drug.get("charOffset").split("-")
                start = int(chars[0])
                end = int(chars[-1]) + 1 #add 1 to convert from inclusive- to exclusive-end index
                label = drug.get("type")
                drugList.append((start, end, label))
            for interaction in sentence.iter("pair"):
                if interaction.get("ddi") == "true":
                    e1 = int(interaction.get("e1").split("e")[-1])
                    e2 = int(interaction.get("e2").split("e")[-1])
                    first = min(e1, e2)
                    second = max(e1, e2)
                    label = interaction.get("type")
                    interactionList.append((first, second, label))
            sentenceList.append((sentenceText, drugList, interactionList))
        docList.append(sentenceList)
    return docList

test = readDrugNER("Test")
print("read test data")

pickle.dump(test, open("Test/TEST", "wb"))
print("dumped test data")
