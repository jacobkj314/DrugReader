import pickle
#import spacy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from numpy import ndarray, array

def main():
    #nlp = spacy.load("en_core_web_lg")

    #labelOptions = ["mechanism", "effect", "advise", "int", "none"]
    labels = list()#this is where we will store the outputted labels

    cols = [str(i) for i in range(300)]#dummy column names

    trainData = pd.DataFrame()#blank 300-column pandas dataframe

    golds: dict[str, list[ndarray]] = pickle.load(open("goldVectors_3-4-peak", "rb"))#load best vectors from phase1
    for label in ["mechanism", "effect", "advise", "int"]:
        print(label)
        gold = golds[label]#select which set of gold vectors to look at
        for vector in gold:
            vector = array([vector])#rotate to row vector
            newData = pd.DataFrame(vector)#create dataFrame
            trainData = pd.concat([trainData, newData], ignore_index = True)#append
            labels.append(label)


    """   for file in os.listdir(docFolder):#get every document file
        with open(os.path.join(docFolder, file), "r") as doc:#open the document
            with open(os.path.join(ansFolder, file + ".key"), "r") as ans:#open the answer key
                doc = nlp(re.sub("\\s+", " ", doc.read()).strip())#pass the raw text into spacy, simplifying whitespace
                ans = ans.read()#get the answer key contents

                answerKey = dict()
                for line in (line.strip() for line in ans.split("\n") if line != ""):#split the document by lines, ignoring blank lines
                    label = line.split(":")[0]#get label
                    if label not in labelOptions:
                        continue
                    answers = [nlp(answer.strip().strip("\"")) for answer in line.split(":")[1].split("/")]#get answers
                    for answer in answers:
                        answerKey[answer] = label


                for token in doc:
                    labels.append("O")#start by assuming its an other
                
                for answer in answerKey.keys():#we need to find where in the document the answer is
                    i = -len(doc)
                    while i <= -len(answer):#try to match the answer to every possible position in the document
                        for j in range(len(answer)):
                            if answer[j].text != doc[i+j].text:#if any of the positions don't match...
                                continue #...leave this loop
                            for k in range(len(answer)):# if it is a full match...
                                labels[i+k] = answerKey[answer]#...we label it
                        i += 1

                newDF = IEutils.convertDocToDict(doc)

                trainData = pd.concat([trainData, newDF], ignore_index = True)"""

    v = DictVectorizer(sparse=False)
    #data = v.fit_transform(trainData.to_dict('records'))
    data = trainData# #I think I can replace the above line with this one

    train = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
    train.fit(data, labels)

    
    pickle.dump((v, train), open("multiclassifier", "wb"))
    

if __name__ == "__main__":
    main()