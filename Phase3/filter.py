#This python script trains the binary filter classifier on extracted extracted dependencies using sklearn. Written by Jacob Johnson for CS6390 at the University of Utah, Spring 2022.
# It is modified from the classifier training script that Lindsay Wilde and I wrote for the final project last semester in CS5340, which in turn was based on the classifier script for the third programming assignment in that class
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from numpy import array
from pandas import DataFrame
from utils import nlp
from spacy.tokens.span import Span


def filter(docs = pickle.load(open("Train/TRAIN", "rb"))):
    good = set()
    bad = set()

    for doc in docs:
        for sentenceText, drugs, interactions in doc:
            sentenceText = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", r"", sentenceText)#remove leading/trailing non-alphanumeric characters
            sentenceAsDoc = nlp(sentenceText)
            drug = [sentenceAsDoc.char_span(start, end) for start, end, _ in drugs]

            #extract POSITIVE examples
            for one, two, _ in interactions:
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
                    pair = depPair(drug[one], drug[two])
                    if pair is not None:
                        good.add(pair)


            #extract NEGATIVE examples
            interactions: list[tuple[int, int]] = [(one, two) for one, two, _ in interactions]
            for one in range(len(drugs)):
                for two in range(one+1, len(drugs)):
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
                                #print("NEGATIVE")
                                pair = depPair(drug[one], drug[two])
                                if pair is not None:
                                    bad.add(pair)

    bad = [b for b in bad if b not in good]#restrict bad contexts to those that have never been good
    good = list(good)

    print(len(good), len(bad))

    train = good + bad
    labels = ["good" for g in good] + ["bad" for b in bad]

    v = DictVectorizer(sparse=False)
    train = v.fit_transform(DataFrame(train).to_dict('records'))
    

    model = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
    model.fit(train, labels)

    return v, model














def depPair(first: Span, second: Span):
    #find peak
    peak = None
    pointer1 = first.root #initialize left side of path
    while peak is None: #wait to find a "lowest common dependency ancestor"
        pointer2 = second.root#initialize right side of path
        while peak is None:
            if pointer1 == pointer2:#when they line up
                peak = pointer1#that's the peak
            if pointer2.head == pointer2:
                break
            pointer2 = pointer2.head #otherwise, iterate right
        if pointer1.head == pointer1:
            break
        pointer1 = pointer1.head #iterate left
    if peak is None:
        raise Exception("No dependency peak found!")
    path = list()
    #calculate first side of path
    pointer = first.root
    while True:
        if pointer == peak:
            peak = len(path)
            path.append(pointer)
            break
        elif pointer == first.root:
            path.append(pointer.dep_)
        else:#somewhere in between
            path.append(pointer)
            path.append(pointer.dep_)
        pointer = pointer.head
    #calculate second side
    pointer = second.root
    while True:
        if pointer == path[peak]:
            break
        elif pointer == second.root:
            path.insert(peak+1, pointer.dep_)
        else:
            path.insert(peak+1, pointer)
            path.insert(peak+1, pointer.dep_)
        pointer = pointer.head

    if peak != 0 and peak != len(path) - 1:
        return (path[peak-1], path[peak+1])









if __name__ == "__main__":
    vModel = filter()
    pickle.dump(vModel, open("crossFilter", "wb"))
    