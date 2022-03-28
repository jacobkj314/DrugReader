#this code extracts vectors from training documents to be used with my system

import pickle
import spacy
import re
from utils import pattern, nlp

def extract(docs = pickle.load(open("Train/TRAIN", "rb"))):
    #accumulators
    golds: dict[str, list] = dict()
    for label in ["mechanism", "effect", "advise", "int"]:
        golds[label] = list() 
    negatives = list() 

    for doc in progressBar(docs, prefix="Extracting Vectors"):
        for sentenceText, drugs, interactions in doc:
            sentenceText = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", r"", sentenceText)#remove leading/trailing non-alphanumeric characters
            sentenceAsDoc = nlp(sentenceText)
            drug = [sentenceAsDoc.char_span(start, end) for start, end, _ in drugs]

            #extract POSITIVE vectors
            for one, two, label in interactions:
                if label not in golds: #This means there is some annotation error, so just skip it
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
                                vector = pattern(drug[one], drug[two])
                                negatives.append(vector)
    return golds, negatives

if __name__ == "__main__":
    g, n = extract()
    pickle.dump(g, open("Phase3/vectors/gold", "wb"))
    pickle.dump(n, open("Phase3/vectors/negative", "wb"))


#code from https://stackoverflow.com/a/34325723
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()