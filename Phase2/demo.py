###This is the demo for my CS6390 Phase II submission, Spring 2022
###Usage: python3 demo.py <filename (use quotes if it includes spaces) OR document text> [-gold <comma-separated gold entities>] [-mclass OR -mbin [resolve] OR -pipe]
import sys
import utils
import re



def __main__():
    #This section of the code gets the document body from command line or a file
    docText = ""
    goldEntities = ""

    try:
        docText = " ".join(sys.argv[1:])

        #handle architecture parameters
        if "-mclass" in sys.argv:
            parts = docText.split("-mclass")
            docText = parts[0].strip()
            utils.detectRelation = utils.detectRelationMulticlass
        elif "-mbin" in sys.argv:
            parts = docText.split("-multimbinbin")
            docText = parts[0].strip()
            utils.detectRelation = utils.detectRelationMultiBinary
            if "ignore" in parts[1]:
                utils.resolveCollisions = False
        elif "-pipe" in sys.argv:
            parts = docText.split("-pipe")
            docText = parts[0].strip()
            utils.detectRelation = utils.detectRelationPipeline

        #handle gold entities
        if "-gold" in sys.argv:
            parts = docText.split("-gold")
            docText = parts[0].strip()
            goldEntities = parts[1].strip()

        #attempt to read file, if needed
        docText = open(sys.argv[1], "r").read()
    except:
        pass

    if goldEntities == "":
        relations = utils.extractRelations(docText)
    else:
        goldList = [drug.strip() for drug in goldEntities.strip().split(",")]

        docWithEntities = list()

        for sentence in utils.nlp(docText).sents:
            sentenceText = sentence.text
            sentenceEntities = list()
            for drug in goldList:
                for match in re.finditer(drug, sentenceText):
                    start, end = match.span()
                    sentenceEntities.append((start, end, "entity")) #user doesn't need to enter what type of entity, because my system doesn't use it anyway
            docWithEntities.append((sentenceText, sentenceEntities))

        relations = utils.extractRelationsFromGoldEntities(docWithEntities)

    #format and print results
    for first, second, relation, sentence in relations: 
        print(first + ", " + second + ": " + relation + " (sentence " + str(sentence) + ")")

if __name__ == "__main__":
    __main__()