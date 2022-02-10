import pickle
import utils
from extract import extractRelations

dev = utils.getGold("Dev")

for doc in dev:
    docText = " ".join([sentenceText for sentenceText, _, _ in doc])
    relations = extractRelations(docText)
