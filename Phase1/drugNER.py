from xml.etree import ElementTree as ET
import spacy
import os
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import pickle
import re

#spacy setup
nlp = spacy.load("en_core_web_lg") 
ner = nlp.get_pipe("ner")

#containers setup
TRAIN_DATA = []
LABELS = ["DRUG", "BRAND", "GROUP", "DRUG_N", "O"]

docFolder = "Train"
for file in os.listdir(docFolder): #get every training document
    if str(file) == "TRAIN":
        continue
    file = os.path.join(docFolder, file)
    #doc = open(file, "r").read()
    root = ET.parse(file).getroot() #parse document as XML tree
    for sentence in root.iter("sentence"):
        sentenceText = sentence.get("text")
        entities = list()#more containers
        entDict = dict()

        for drug in sentence.iter("entity"):
            chars = drug.get("charOffset").split("-")
            start = int(chars[0])
            end = int(chars[-1]) + 1 #add 1 to convert from inclusive- to exclusive-end index
            label = drug.get("type")
            overlapping = False
            for s,e,_ in entities : 
                if not (start < s and end <= s or start >= e and end > e) :
                    overlapping = True
                    print("overlapping")
                    break
            if not overlapping : 
                entities.append((start, end, label))
        entDict["entities"] = entities
        TRAIN_DATA.append((sentenceText, entDict))

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
  for iteration in range(25):
    print(iteration)
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