import os
from xml.etree import ElementTree as ET
import pickle

def readDrugNER(docFolder : str) -> list[set[tuple[int, int, str]]]:
    sentenceList = list()
    for file in os.listdir(docFolder): #get every training document
        if str(file) == "TEST":#ignore pickled file
            continue
        file = os.path.join(docFolder, file)
        root = ET.parse(file).getroot() #parse document as XML tree
        for sentence in root.iter("sentence"):
            drugSet = set()
            for drug in sentence.iter("entity"):
                chars = drug.get("charOffset").split("-")
                start = int(chars[0])
                end = int(chars[-1]) + 1 #add 1 to convert from inclusive- to exclusive-end index
                label = drug.get("type")
                drugSet.add((start, end, label))
            sentenceList.append(drugSet)
    return sentenceList

train = readDrugNER("Test")
print("read test data")

pickle.dump(train, open("Test/TEST", "wb"))
print("dumped test data")
