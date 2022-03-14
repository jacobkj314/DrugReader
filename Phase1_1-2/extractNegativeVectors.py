#this code extracts and "pickles" all the gold vectors from training documents to be used with my system
import pickle
import spacy
import re
from utils import getGold, extractPattern

partition = "Train"

gold = getGold(partition)

nlp = spacy.load("en_core_web_lg")


patterns = list() 

for doc in gold:
    for sentenceText, drugs, interactions in doc:
        sentenceText = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", r"", sentenceText)#remove leading/trailing non-alphanumeric characters
        #if len(interactions) != 0:
            #print("\n" + sentenceText + ":")
        sentenceAsDoc = nlp(sentenceText)
        drug = [sentenceAsDoc.char_span(start, end) for start, end, _ in drugs]
        drugStr = [d.text for d in drug if d is not None]#drugs as text for creating vectors
        golds: list[tuple[int, int]] = [(one, two) for one, two, _ in interactions]
        #start modifications
        for one in range(len(drugs)):
            for two in range(len(drugs)):
                if(one != two and (one, two) not in golds):
                    #some entities are parsed by spacy into different word boundaries. This check makes sure that we extract gold patterns from entities that spacy can detect
                    if drug[one] is None or drug[two] is None:
                        continue
                    #some sentences are parsed by spacy into two sentences. This loop makes sure that we use the part of the sentence that includes both entities, if possible
                    sentence = None
                    for s in sentenceAsDoc.sents:
                        if drug[one].root in s and drug[two].root in s:
                            sentence = s
                            break
                    if sentence is not None:#we can extract!
                        #print(drug[one].text + ", " + drug[two].text + ": " + extractPattern(drug[one], drug[two], sentence))
                        pattern = extractPattern(drug[one], drug[two], sentence, drugStr)
                        #print(pattern[0])
                        patterns.append(pattern)

pickle.dump(patterns, open("negativeVectors-peak-minus", "wb"))