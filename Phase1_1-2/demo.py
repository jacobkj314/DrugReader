import sys
from utils import nlp, extractRelations, extractRelationsFromGoldEntities
import re



def __main__():
    #This section of the code gets the document body from command line or a file
    docText = ""
    goldEntities = ""

    try:
        docText = " ".join(sys.argv[1:])

        if "-gold" in sys.argv:
            parts = docText.split("-gold")
            docText = parts[0].strip()
            goldEntities = parts[1].strip()

        docText = open(sys.argv[1], "r").read()
    except:
        pass

    if goldEntities == "":
        relations = extractRelations(docText)
    else:
        goldList = [drug.strip() for drug in goldEntities.strip().split(",")]

        docWithEntities = list()

        for sentence in nlp(docText).sents:
            sentenceText = sentence.text
            sentenceEntities = list()
            for drug in goldList:
                for match in re.finditer(drug, sentenceText):
                    start, end = match.span()
                    sentenceEntities.append((start, end, "entity")) #user doesn't need to enter what type of entity, because my system doesn't use it anyway
            docWithEntities.append((sentenceText, sentenceEntities))

        relations = extractRelationsFromGoldEntities(docWithEntities)

    #format and print results
    for first, second, relation, sentence in relations: 
        print(first + ", " + second + ": " + relation + " (sentence " + str(sentence) + ")")

if __name__ == "__main__":
    __main__()