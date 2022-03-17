#This python script trains the pipeline classifiers on extracted context vectors using sklearn. Written by Jacob Johnson for CS6390 at the University of Utah, Spring 2022.
# It is modified from the classifier training script that Lindsay Wilde and I wrote for the final project last semester in CS5340, which in turn was based on the classifier script for the third programming assignment in that class
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from numpy import ndarray, array

def main():
    mainLabels = list()#this is where we will store the outputted labels
    multiLabels = list()

    trainData = pd.DataFrame()#blank 300-column pandas dataframe

    #train on gold vectors
    golds: dict[str, list[ndarray]] = pickle.load(open("goldVectors-peak", "rb"))#load best vectors from phase1
    for label in ["mechanism", "effect", "advise", "int"]:
        gold = golds[label]#select which set of gold vectors to look at
        print(f"{label} ({len(gold)})")
        for vector in gold:
            trainData = pd.concat([trainData, vector], ignore_index = True)#append
            mainLabels.append("true")
            multiLabels.append(label)

    #train and save multimodel
    data = trainData# #I think I can replace the above line with this one

    train = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
    train.fit(data, multiLabels)
    
    pickle.dump(train, open("pipeline-multi", "wb"))


    #train on negative vectors
    negatives = pickle.load(open("negativeVectors-peak", "rb"))
    print(len(negatives))
    for vector in negatives[::(49514//4311)]:#get as many negatives as there are total positives, as determined by devset experiments
        trainData = pd.concat([trainData, vector], ignore_index = True)#append
        mainLabels.append("false")
        

    #v = DictVectorizer(sparse=False)
    #data = v.fit_transform(trainData.to_dict('records'))
    data = trainData# #I think I can replace the above line with this one

    train = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
    train.fit(data, mainLabels)

    
    pickle.dump(train, open("pipeline-main", "wb"))

















    
    

if __name__ == "__main__":
    main()