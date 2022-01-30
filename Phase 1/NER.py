import spacy
import os
import re
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import pickle
import sys

#These are some tutorials we found that helped us in setting up this Spacy model:
#https://www.youtube.com/watch?v=9mXoGxAn6pM&ab_channel=DeepakJohnReji 
#https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
#https://stackoverflow.com/questions/66675261/how-can-i-work-with-example-for-nlp-update-problem-with-spacy3-0


nlp = spacy.load("en_core_web_lg") 
ner = nlp.get_pipe("ner")


#docFolder = sys.argv[1] #docs
#ansFolder = sys.argv[2] #anskeys
docFolder = "../Train"

#labels = list()#this is where we will store the outputted labels

TRAIN_DATA = []
#LABELS = ["ACQUIRED", "ACQBUS", "ACQLOC", "DLRAMT", "PURCHASER", "SELLER", "STATUS"]
LABELS = []

for file in os.listdir(docFolder):#get every document file
    with open(os.path.join(docFolder, file), "r") as doc:#open the document
        with open(os.path.join(ansFolder, file + ".key"), "r") as ans:#open the answer key
            doc = re.sub("\\s+", " ", doc.read()).strip() #pass the raw text into spacy, simplifying whitespace
            ans = ans.read()#get the answer key contents

            entities = list()
            entDict = dict()

            for line in (line.strip() for line in ans.split("\n") if line != ""):#split the document by lines, ignoring blank lines
                label = line.split(":")[0]#get label
                answers = [answer.strip().strip("\"") for answer in line.split(":")[1].split("/")]#get answers
                for answer in answers:
                    for m in re.finditer(answer, doc) :
                        start, end = m.start(0), m.end(0)
                        overlapping = False
                        for s,e,_ in entities : 
                            if not (start < s and end <= s or start >= e and end > e) :
                                overlapping = True
                                break
                        if not overlapping : 
                            entities.append((start, end, label))
                    #entities.extend([(m.start(0), m.end(0), label) for m in re.finditer(answer, doc)])
            entDict["entities"] = entities

            TRAIN_DATA.append((doc, entDict))


for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Add the new label to ner
for LABEL in LABELS :
    ner.add_label(LABEL)

# Resume training
optimizer = nlp.resume_training()
move_names = list(ner.move_names)

# Disable pipeline components you dont need to change
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

  # Training
  for iteration in range(40):

    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    # batch up the examples using spaCy's minibatch
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        for text, annotations in batch:
            # create Example
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations) #help from https://stackoverflow.com/questions/66675261/how-can-i-work-with-example-for-nlp-update-problem-with-spacy3-0
            # Update the model
            nlp.update([example], losses=losses, drop=0.3)
            # # # print("Losses", losses)


pickle.dump(nlp, open("NER", "wb"))

