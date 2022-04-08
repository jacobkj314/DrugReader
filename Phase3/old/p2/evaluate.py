###This is the evaluation code for my CS6390 Phase II submission, Spring 2022
###Usage: python3 evaluate.py  [-test (to use testset data rather than devset data)] [-goldIgnore (to use spaCy-predicted entity mentions)] [-mclass OR -mbin OR -mbinResolve OR -pipe (to select classifier layout)]

import sys
import pickle
import utils
from trainMulticlass import classifier
import vectors
from statistics import mean
from numpy import concatenate

def evaluate(partition, useGold = True, classifier = None):
    #overwrite which classifier to use for this session
    utils.multiclassifier = classifier

    #accumulator variables for answers, guesses, and hits in each category
    mechanismAnswers = 0; mechanismGuesses = 0; mechanismHits = 0
    effectAnswers = 0; effectGuesses = 0; effectHits = 0
    adviseAnswers = 0; adviseGuesses = 0; adviseHits = 0
    intAnswers = 0; intGuesses = 0; intHits = 0

    for i, doc in enumerate(partition):
        print(i)#print which document we are on
        actualRelations = set()
        for sentenceText, drugChars, interactions in doc:
            drugs = [sentenceText[start:end] for start, end, _ in drugChars]
            interactions = [(utils.MentionPair(drugs[first], drugs[second]), label) for first, second, label in interactions]
            for interaction in interactions:
                actualRelations.add(interaction)

        if useGold:
            docWithEntities = [(sentenceText, drugChars) for sentenceText, drugChars, _ in doc]
            extractedRelations = {(utils.MentionPair(first, second), label) for first, second, label, sentence in utils.extractRelationsFromGoldEntities(docWithEntities)} #use if providing gold entites
        else:
            docText = " ".join([sentenceText for sentenceText, _, _ in doc])#assemble document - use if not providing gold entities
            extractedRelations = {(pair, label) for pair, label in utils.extractRelations(docText)} #use if not providing gold entities

        print(extractedRelations, "|", actualRelations)

        #divide actual relations by label
        actualMechanisms = {relation for relation in actualRelations if relation[1] == "mechanism"}
        actualEffects = {relation for relation in actualRelations if relation[1] == "effect"}
        actualAdvises = {relation for relation in actualRelations if relation[1] == "advise"}
        actualInts = {relation for relation in actualRelations if relation[1] == "int"}

        #divide extracted relations by label
        extractedMechanisms = {relation for relation in extractedRelations if relation[1] == "mechanism"}
        extractedEffects = {relation for relation in extractedRelations if relation[1] == "effect"}
        extractedAdvises = {relation for relation in extractedRelations if relation[1] == "advise"}
        extractedInts = {relation for relation in extractedRelations if relation[1] == "int"}

        #update number of correct answers in each category
        mechanismAnswers += len(actualMechanisms)
        effectAnswers += len(actualEffects)
        adviseAnswers += len(actualAdvises)
        intAnswers += len(actualInts)

        #update number of system guesses in each category
        mechanismGuesses += len(extractedMechanisms)
        effectGuesses += len(extractedEffects)
        adviseGuesses += len(extractedAdvises)
        intGuesses += len(extractedInts)

        #update number of system hits in each category
        mechanismHits += len(actualMechanisms.intersection(extractedMechanisms))
        effectHits += len(actualEffects.intersection(extractedEffects))
        adviseHits += len(actualAdvises.intersection(extractedAdvises))
        intHits += len(actualInts.intersection(extractedInts))

    return (mechanismAnswers, effectAnswers, adviseAnswers, intAnswers, mechanismGuesses, effectGuesses, adviseGuesses, intGuesses, mechanismHits, effectHits, adviseHits, intHits)


def analyze(results):
    (mechanismAnswers, effectAnswers, adviseAnswers, intAnswers, mechanismGuesses, effectGuesses, adviseGuesses, intGuesses, mechanismHits, effectHits, adviseHits, intHits) = results

    #sentence-level evaluations:
    totalAnswers = mechanismAnswers + effectAnswers + adviseAnswers + intAnswers
    totalGuesses = mechanismGuesses + effectGuesses + adviseGuesses + intGuesses
    totalHits = mechanismHits + effectHits + adviseHits + intHits

    mechanismPrecision = mechanismHits/mechanismGuesses if mechanismGuesses != 0 else 0
    effectPrecision = effectHits/effectGuesses if effectGuesses != 0 else 0
    advisePrecision = adviseHits/adviseGuesses if adviseGuesses != 0 else 0
    intPrecision = intHits/intGuesses if intGuesses != 0 else 0
    totalPrecision = totalHits/totalGuesses if totalGuesses != 0 else 0

    mechanismRecall = mechanismHits/mechanismAnswers if mechanismAnswers != 0 else 1
    effectRecall = effectHits/effectAnswers if effectAnswers != 0 else 1
    adviseRecall = adviseHits/adviseAnswers if adviseAnswers != 0 else 1
    intRecall = intHits/intAnswers if intAnswers != 0 else 1
    totalRecall = totalHits/totalAnswers if totalAnswers != 0 else 1

    mechanismF = (2*mechanismPrecision*mechanismRecall)/(mechanismPrecision + mechanismRecall) if (mechanismPrecision + mechanismRecall) != 0 else 0
    effectF = (2*effectPrecision*effectRecall)/(effectPrecision + effectRecall) if (effectPrecision + effectRecall) != 0 else 0
    adviseF = (2*advisePrecision*adviseRecall)/(advisePrecision + adviseRecall) if (advisePrecision + adviseRecall) != 0 else 0
    intF = (2*intPrecision*intRecall)/(intPrecision + intRecall) if (intPrecision + intRecall) != 0 else 0
    totalF = (2*totalPrecision*totalRecall)/(totalPrecision + totalRecall) if (totalPrecision + totalRecall) != 0 else 0

    return (mechanismPrecision, mechanismRecall, mechanismF, effectPrecision, effectRecall, effectF, advisePrecision, adviseRecall, adviseF, intPrecision, intRecall, intF, totalPrecision, totalRecall, totalF)



#helper method for ablation testing
boundaries = [0, 1, 301, 901, 1201, 1801, 2101, 2102, 2103, 2104, 2404, 2704, 3304, 3305]
def project(vector, featureSet):
    result = []
    for i, feature in enumerate(featureSet):
        if feature:
            result = concatenate((result, vector[boundaries[i]:boundaries[i+1]]))
    return result










def main():
    """if "-goldIgnore" in sys.argv:
        useGold = False"""

    results = list()

    if True:# #"-test" in sys.argv:
        results = list()

        data = pickle.load(open("../../../crossValidate", "rb"))
        for fold in range(5):
            train = data[:200*fold] + data[200*(fold+1):]
            test = data[200*fold:200*(fold+1)]

            #utils.vectorizer, utils.filter = filter.filter(train)#train filter
            g, n = vectors.extract(train)#extract vectors from train
            c = classifier(g, n)#build classifier from train vectors
            results.append(evaluate(classifier = c, partition = test))#evaluate on test partition
            
        for result in results:
            print(result)

        results = tuple((mean([results[result][parameter] for result in results]) for parameter in range(12)))#average results
        results = analyze(results)#analyze results

        print("mechP\tmechR\tmechF\teffP\teffR\teffF\tadvP\tadvR\tadvF\tintP\tintR\tintF\ttotP\ttotR\ttotF")
        for r in results:
            print(r, end = "\t")








if __name__ == "__main__":
    main()