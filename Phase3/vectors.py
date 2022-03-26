#this code extracts vectors from training documents to be used with my system

import pickle
import spacy
import re
from utils import pattern, nlp

def extract(docs = pickle.load(open("Train/TRAIN", "rb"))):
    #accumulators
    golds: dict[str, list] = dict()
    for label in doc:
        golds[label] = list() 
    negatives = list() 

    for doc in docs:
        for sentenceText, drugs, interactions in doc:
            sentenceText = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", r"", sentenceText)#remove leading/trailing non-alphanumeric characters
            sentenceAsDoc = nlp(sentenceText)
            drug = [sentenceAsDoc.char_span(start, end) for start, end, _ in drugs]

            #extract POSITIVE vectors
            for one, two, label in interactions:
                if label not in doc: #This means there is some annotation error, so just skip it
                    continue
                #some entities are parsed by spacy into different word boundaries. This check makes sure that we extract gold patterns from entities that spacy can detect
                if drug[one] is None or drug[two] is None:
                    continue
                #some sentences are parsed by spacy into two sentences. This loop makes sure that we use the part of the sentence that includes both entities, if possible
                sentence = None
                for s in sentenceAsDoc.sents:
                    if drug[one].sent == s and drug[two].sent == s:
                        sentence = s
                        break
                if sentence is not None:#we can extract!
                    vector = pattern(drug[one], drug[two])
                    print(vector[1])
                    golds[label].append(vector)

            #extract NEGATIVE vectors
            interactions: list[tuple[int, int]] = [(one, two) for one, two, _ in interactions]
            for one in range(len(drugs)):
                for two in range(len(drugs)):
                    if one != two:
                        if (one, two) not in interactions:
                            #some entities are parsed by spacy into different word boundaries. This check makes sure that we extract gold patterns from entities that spacy can detect
                            if drug[one] is None or drug[two] is None:
                                continue
                            #some sentences are parsed by spacy into two sentences. This loop makes sure that we use the part of the sentence that includes both entities, if possible
                            sentence = None
                            for s in sentenceAsDoc.sents:
                                if drug[one].sent == s and drug[two].sent == s:
                                    sentence = s
                                    break
                            if sentence is not None:#we can extract!
                                #print(drug[one].text + ", " + drug[two].text + ": " + extractPattern(drug[one], drug[two], sentence))
                                vector = pattern(drug[one], drug[two])
                                print(vector[1])
                                negatives.append(vector)
    return golds, negatives

if __name__ == "__main__":
    g, n = extract()
    pickle.dump(g, open("Phase3/vectors/gold", "wb"))
    pickle.dump(n, open("Phase3/vectors/negative", "wb"))