import pickle
import spacy
import re
from utils import getGold, extractPattern

partition = "Train"

gold = getGold(partition)

nlp = spacy.load("en_core_web_lg")

labels = dict()
labels["effect"] = set()
labels["mechanism"] = set()
labels["advice"] = set()
labels["int"] = set()

for doc in gold:
    for sentenceText, drugs, interactions in doc:
        sentenceText = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", r"", sentenceText)#remove leading/trailing non-alphanumeric characters
        if len(interactions) != 0:
            print("\n" + sentenceText + ":")
        sentence = nlp(sentenceText)[:]
        for first, second, label in interactions:
            drug1 = sentence.char_span(drugs[first][0], drugs[first][1])
            drug2 = sentence.char_span(drugs[second][0], drugs[second][1])
            if drug1 is None:
                print("drug1: " + sentenceText[drugs[first][0]: drugs[first][1]])
            if drug2 is None:
                print("drug2: " + sentenceText[drugs[second][0]: drugs[second][1]])
            print(drug1.text + ", " + drug2.text + ": " + extractPattern(drug1, drug2, sentence))