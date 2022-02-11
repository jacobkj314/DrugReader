import sys
from utils import ner, detectRelation

def extractRelations(docText: str) -> list[tuple[str, str, str, int]]:
    #the document is passed into the spacy model to extract drug entities
    doc = ner(docText)
    #extract all possible relations
    relations = list()#a container to hold extracted relations
    for s, sentence in enumerate(doc.sents):#the document is split into sentences
        print(sentence)
        ents = sentence.ents#spacy extracts the entities
        entCount = len(ents)
        for i, first in enumerate(ents):
            for second in [ents[j] for j in range(i + 1, entCount, 1)]:
                relation = detectRelation(first, second, sentence)#each potential pair is compared
                if relation is not None:
                    relations.append((first.text, second.text, relation, s))#if there is a relation, add it to the extracted relations
    return relations

def __main__():
    #This section of the code gets the document body from command line or a file
    docText = ""
    try:
        docText = " ".join(sys.argv[1:])
        docText = open(sys.argv[1], "r").read()
    except:
        pass

    relations = extractRelations(docText)

    #format and print results
    for first, second, relation, sentence in relations: 
        print(first + ", " + second + ": " + relation + " (sentence " + str(sentence) + ")")

if __name__ == "__main__":
    __main__()