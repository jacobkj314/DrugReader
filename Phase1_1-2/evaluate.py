import utils

test = utils.getGold("Dev")

useGold = True

def evaluate():

    #accumulator variables for answers, guesses, and hits in each category
    mechanismAnswers = 0
    mechanismGuesses = 0
    mechanismHits = 0

    effectAnswers = 0
    effectGuesses = 0
    effectHits = 0

    adviseAnswers = 0
    adviseGuesses = 0
    adviseHits = 0

    intAnswers = 0
    intGuesses = 0
    intHits = 0

    for i, doc in enumerate(test):
        print(i)#print which document we are on
        actualRelations = set()
        for sentenceText, drugChars, interactions in doc:
            drugs = [sentenceText[start:end] for start, end, _ in drugChars]
            interactions = [(drugs[first], drugs[second], label) for first, second, label in interactions]
            for interaction in interactions:
                actualRelations.add(interaction)
        
        if useGold:
            docWithEntities = [(sentenceText, drugChars) for sentenceText, drugChars, _ in doc]
            extractedRelations = {(first, second, label) for first, second, label, _ in utils.extractRelationsFromGoldEntities(docWithEntities)} #use if providing gold entites
        else:
            docText = " ".join([sentenceText for sentenceText, _, _ in doc])#assemble document - use if not providing gold entities
            extractedRelations = {(first, second, label) for first, second, label, _ in utils.extractRelations(docText)} #use if not providing gold entities

        print(extractedRelations, "|", actualRelations)

        #divide actual relations by label
        actualMechanisms = {relation for relation in actualRelations if relation[2] == "mechanism"}
        actualEffects = {relation for relation in actualRelations if relation[2] == "effect"}
        actualAdvises = {relation for relation in actualRelations if relation[2] == "advise"}
        actualInts = {relation for relation in actualRelations if relation[2] == "int"}

        #divide extracted relations by label
        extractedMechanisms = {relation for relation in extractedRelations if relation[2] == "mechanism"}
        extractedEffects = {relation for relation in extractedRelations if relation[2] == "effect"}
        extractedAdvises = {relation for relation in extractedRelations if relation[2] == "advise"}
        extractedInts = {relation for relation in extractedRelations if relation[2] == "int"}

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

    
    print("\tmechP\tmechR\tmechF\teffP\teffR\teffF\tadvP\tadvR\tadvF\tintP\tintR\tintF\ttotP\ttotR\ttotF")
    print(f"S:\t\t{mechanismPrecision}\t\t{mechanismRecall}\t\t{mechanismF}\t\t{effectPrecision}\t\t{effectRecall}\t\t{effectF}\t\t{advisePrecision}\t\t{adviseRecall}\t\t{adviseF}\t\t{intPrecision}\t\t{intRecall}\t\t{intF}\t\t{totalPrecision}\t\t{totalRecall}\t\t{totalF}")

if __name__ == "__main__":
    evaluate()