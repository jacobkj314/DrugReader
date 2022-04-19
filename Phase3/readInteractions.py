from xml.etree import ElementTree as ET
import sys

def readInteractions(file): 
    root = ET.parse(file).getroot() #parse document as XML tree
    sentenceList = list()
    textList = list()
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
                textList.append((sentenceText[drugList[e1][0]:drugList[e1][1]], sentenceText[drugList[e2][0]:drugList[e2][1]], label))# #

        sentenceList.append((sentenceText, drugList, interactionList))
    return textList

def main():
    for interaction in readInteractions(sys.argv[1]):
        print(interaction)

if __name__ == "__main__":
    main()
