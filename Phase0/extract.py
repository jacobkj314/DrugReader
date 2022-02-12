import sys
from utils import extractRelations



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