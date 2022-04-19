###This is the demo for my CS6390 Phase III submission, Spring 2022
###Usage: python3 demo.py <filename (use quotes if it includes spaces) OR document text> [-gold <comma-separated gold entities>] [-features <comma-separated features to use>]
import sys
import utils
import re



def __main__():
    #This section of the code gets the document body from command line or a file
    docText = ""
    goldEntities = ""
    features = ""

    try:
        docText = " ".join(sys.argv[1:])

        #handle feature selection
        if "-features" in sys.argv:
            parts = docText.split("-features")
            docText = parts[0].strip()
            features = parts[1].strip()

        #handle gold entities
        if "-gold" in sys.argv:
            parts = docText.split("-gold")
            docText = parts[0].strip()
            goldEntities = parts[1].strip()

        #attempt to read file, if needed
        docText = open(sys.argv[1], "r").read()
    except:
        pass

    if features != "":#if the user supplied custom features, we will have to retrain the model
        features = [feature.strip() for feature in features.strip().split(",")]
        utils.features = [True if f in features else False for f in ["path1len", "path1", "dep1", "peak", "dep2", "path2", "path2len", "swapped", "surfdist", "root", "rootPath", "rootDep", "rootLen"]] 
        from vectors import extract
        import pickle
        g, n = extract(pickle.load(open("docs/crossValidate", "rb")))
        from classifier import classifier
        utils.classifier = classifier(1, g, n)

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
    for pair, label in relations: 
        print(str(pair) + ": " + label)

if __name__ == "__main__":
    __main__()
