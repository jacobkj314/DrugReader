import predict
from spacy import displacy

ner = predict.getNER()

displacy.serve(predict.ner(), style = "dep")