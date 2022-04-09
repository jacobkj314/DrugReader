###This is the evaluation code for my CS6390 Phase II submission, Spring 2022
###Usage: python3 evaluate.py  [-test (to use testset data rather than devset data)] [-goldIgnore (to use spaCy-predicted entity mentions)] [-mclass OR -mbin OR -mbinResolve OR -pipe (to select classifier layout)]

import sys
import pickle
import evalFilterUtils
import classifier
import vectors
import filter
from statistics import mean
from numpy import concatenate

def evaluate(partition):

    #accumulator variables for answers, guesses, and hits 
    answers = 0; guesses = 0; hits = 0

    for i, doc in enumerate(partition):
        print(i)#print which document we are on
        actualRelations = set()
        for sentenceText, drugChars, interactions in doc:
            drugs = [sentenceText[start:end] for start, end, _ in drugChars]
            interactions = [evalFilterUtils.MentionPair(drugs[first], drugs[second]) for first, second, label in interactions]
            for interaction in interactions:
                actualRelations.add(interaction)

        docWithEntities = [(sentenceText, drugChars) for sentenceText, drugChars, _ in doc]
        extractedRelations = {pair for pair, label in evalFilterUtils.extractRelationsFromGoldEntities(docWithEntities)} #use if providing gold entites
    
        print(extractedRelations, "|", actualRelations)

        answers += len(actualRelations)
        guesses += len(extractedRelations)
        hits += len(actualRelations.intersection(extractedRelations))

    return (answers, guesses, hits)

def analyze(results):
    (totalAnswers, totalGuesses, totalHits) = results

    totalPrecision = totalHits/totalGuesses if totalGuesses != 0 else 0
    totalRecall = totalHits/totalAnswers if totalAnswers != 0 else 1
    totalF = (2*totalPrecision*totalRecall)/(totalPrecision + totalRecall) if (totalPrecision + totalRecall) != 0 else 0

    return (totalPrecision, totalRecall, totalF)










def main():

    results = list()
    pairs = list()

    if True:# #"-test" in sys.argv:
        data = pickle.load(open("crossValidate", "rb"))
        for fold in range(5):
            evalFilterUtils.pairs = 0
            train = data[:200*fold] + data[200*(fold+1):]
            test = data[200*fold:200*(fold+1)]

            evalFilterUtils.vectorizer, evalFilterUtils.filter = filter.filter(train)#train filter

            results.append(evaluate(partition = test))#evaluate on test partition
            pairs.append(evalFilterUtils.pairs)
        
        for result in results: 
            print(result)
        
        results = tuple((sum([row[col] for row in results]) for col in range(3)))#aggregate results from folds

        results = analyze(results)#analyze results

        for r in results:
            print(r, end = "\t")
        
        print()
        print(pairs)


   



if __name__ == "__main__":
    main()