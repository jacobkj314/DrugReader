#This python script trains the multiclassifer on extracted context vectors using sklearn. 
# Written by Jacob Johnson for CS6390 at the University of Utah, Spring 2022.
# It is modified from the classifier training script that Lindsay Wilde and I wrote for the final project last semester in CS5340, which in turn was based on the classifier script for the third programming assignment in that class
import pickle
from sklearn.linear_model import LogisticRegression
from numpy import array, ndarray, concatenate

def classifier(ratio, golds, negatives):
    labels = list();data = list()#ndarray((0,2103))#accumulators
    #train on gold vectors
    for label in golds:
        for vector in golds[label]:
            #data = concatenate((data, (vector,)))#append vector
            data.append(vector)
            labels.append(label)
    #train on negative vectors
    for vector in negatives[::int(len(negatives)/sum((len(golds[label]) for label in golds))/ratio)]:
        #data = concatenate((data, (vector,)))#append vector
        data.append(vector)
        labels.append("none")

    data = array(data)
    model = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
    model.fit(data, labels)
    return model

if __name__ == "__main__":
    pickle.dump(classifier(), open("Phase3/classifier", "wb"))