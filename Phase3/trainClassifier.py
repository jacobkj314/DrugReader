#This python script trains the multiclassifer on extracted context vectors using sklearn. Written by Jacob Johnson for CS6390 at the University of Utah, Spring 2022.
# It is modified from the classifier training script that Lindsay Wilde and I wrote for the final project last semester in CS5340, which in turn was based on the classifier script for the third programming assignment in that class
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from numpy import ndarray, concatenate

def main():
    labels = list()#this is where we will store the outputted labels

    trainData = ndarray((0,2103))

    #train on gold vectors
    golds: dict[str, list[ndarray]] = pickle.load(open("Phase3/vectors/gold", "rb"))#load best vectors from phase1
    for label in ["mechanism", "effect", "advise", "int"]:
        gold = golds[label]#select which set of gold vectors to look at
        print(f"{label} ({len(gold)})")
        for vector in gold:
            #vector = array([vector])#rotate to row vector
            #newData = pd.DataFrame(vector)#create dataFrame
            trainData = concatenate((trainData, (vector,)))#append
            labels.append(label)

    #"""
    #train on negative vectors
    negatives = pickle.load(open("Phase3/vectors/negative", "rb"))
    print(len(negatives))
    for vector in negatives[::(49514//4311)]:
        #vector = array([vector])#rotate to row vector
        #newData = pd.DataFrame(vector)#create dataFrame
        trainData = concatenate((trainData, (vector,)))#append
        labels.append("none")
    #""" 

    #v = DictVectorizer(sparse=False)
    #data = v.fit_transform(trainData.to_dict('records'))
    data = trainData# #I think I can replace the above line with this one

    train = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
    train.fit(data, labels)

    
    pickle.dump(train, open("Phase3/classifier", "wb"))
    

if __name__ == "__main__":
    main()