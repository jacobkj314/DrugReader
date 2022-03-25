#This python script trains the multiclassifer on extracted context vectors using sklearn. Written by Jacob Johnson for CS6390 at the University of Utah, Spring 2022.
# It is modified from the classifier training script that Lindsay Wilde and I wrote for the final project last semester in CS5340, which in turn was based on the classifier script for the third programming assignment in that class
import pickle
from sklearn.linear_model import LogisticRegression
from numpy import ndarray, concatenate

#load vectors
golds: dict[str, list[ndarray]] = pickle.load(open("Phase3/vectors/gold", "rb"))
negatives = pickle.load(open("Phase3/vectors/negative", "rb"))

def classifier(ratio = 1.0):
    labels = list();data = ndarray((0,2103))#accumulators
    #train on gold vectors
    for label in golds:
        for vector in golds[label]:
            data = concatenate((data, (vector,)))#append vector
            labels.append(label)
    #train on negative vectors
    for vector in negatives[::int(49514/4311/ratio)]:
        data = concatenate((data, (vector,)))#append vector
        labels.append("none")
    model = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
    model.fit(data, labels)
    return model

if __name__ == "__main__":
    pickle.dump(classifier(), open("Phase3/classifier", "wb"))