import utils

test = utils.getGold("Test")

#added this part to the phase1 evaluation script to make the results compatible with those of phase3
class MentionPair:
    def __init__(self, one, two):
        self.one = one
        self.two = two
    def __eq__(self, other):
        if type(other) is not MentionPair:
            return False
        return {str(self.one).lower(), str(self.two).lower()} == {str(other.one).lower(), str(other.two).lower()}
    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash(tuple(sorted([str(self.one).lower(), str(self.two).lower()])))
    def __str__(self):
        return str(self())
    def __repr__(self):
        return repr(self())
    def __getitem__(self, key):
        return self().__getitem__(key)
    def __iter__(self):
        return self().__iter__()
    def __call__(self):
        return (self.one, self.two)


def evaluate():

    #hold all entities extracted by system
    cumulativeActualMechanisms = set()
    cumulativeActualEffects = set()
    cumulativeActualAdvises = set()
    cumulativeActualInts = set()

    #hold all entities extracted by system
    cumulativeExtractedMechanisms = set()
    cumulativeExtractedEffects = set()
    cumulativeExtractedAdvises = set()
    cumulativeExtractedInts = set()

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
        
        docWithEntities = [(sentenceText, drugChars) for sentenceText, drugChars, _ in doc]
        extractedRelations = {(first, second, label) for first, second, label, _ in utils.extractRelationsFromGoldEntities(docWithEntities)} #use if providing gold entites
        
        #docText = " ".join([sentenceText for sentenceText, _, _ in doc])#assemble document - use if not providing gold entities
        #extractedRelations = {(first, second, label) for first, second, label, _ in utils.extractRelations(docText)} #use if not providing gold entities



        #this is where the output is made compatible with phase3, also changing 2s to 1s when dividing relations by label # #
        actualRelations = {(MentionPair(first, second), label) for first, second, label in actualRelations}
        extractedRelations = {(MentionPair(first, second), label) for first, second, label in extractedRelations}


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

        #add actual relations in cumulative sets
        cumulativeActualMechanisms.update(actualMechanisms)
        cumulativeActualEffects.update(actualEffects)
        cumulativeActualAdvises.update(actualAdvises)
        cumulativeActualInts.update(actualInts)

        #add extracted relations in cumulative sets
        cumulativeExtractedMechanisms.update(extractedMechanisms)
        cumulativeExtractedEffects.update(extractedEffects)
        cumulativeExtractedAdvises.update(extractedAdvises)
        cumulativeExtractedInts.update(extractedInts)

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

    #cumulative evaluations
    cumulativeMechanismAnswers = len(cumulativeActualMechanisms)
    cumulativeEffectAnswers = len(cumulativeActualEffects)
    cumulativeAdviseAnswers = len(cumulativeActualAdvises)
    cumulativeIntAnswers = len(cumulativeActualInts)
    cumulativeTotalAnswers = cumulativeMechanismAnswers + cumulativeEffectAnswers + cumulativeAdviseAnswers + cumulativeIntAnswers

    cumulativeMechanismGuesses = len(cumulativeExtractedMechanisms)
    cumulativeEffectGuesses = len(cumulativeExtractedEffects)
    cumulativeAdviseGuesses = len(cumulativeExtractedAdvises)
    cumulativeIntGuesses = len(cumulativeExtractedInts)
    cumulativeTotalGuesses = cumulativeMechanismGuesses + cumulativeEffectGuesses + cumulativeAdviseGuesses + cumulativeIntGuesses

    cumulativeMechanismHits = len(cumulativeActualMechanisms.intersection(cumulativeExtractedMechanisms))
    cumulativeEffectHits = len(cumulativeActualEffects.intersection(cumulativeExtractedEffects))
    cumulativeAdviseHits = len(cumulativeActualAdvises.intersection(cumulativeExtractedAdvises))
    cumulativeIntHits = len(cumulativeActualInts.intersection(cumulativeExtractedInts))
    cumulativeTotalHits = cumulativeMechanismHits + cumulativeEffectHits + cumulativeAdviseHits + cumulativeIntHits

    cumulativeMechanismPrecision = cumulativeMechanismHits/cumulativeMechanismGuesses if cumulativeMechanismGuesses != 0 else 0
    cumulativeEffectPrecision = cumulativeEffectHits/cumulativeEffectGuesses if cumulativeEffectGuesses != 0 else 0
    cumulativeAdvisePrecision = cumulativeAdviseHits/cumulativeAdviseGuesses if cumulativeAdviseGuesses != 0 else 0
    cumulativeIntPrecision = cumulativeIntHits/cumulativeIntGuesses if cumulativeIntGuesses != 0 else 0
    cumulativeTotalPrecision = cumulativeTotalHits/cumulativeTotalGuesses if cumulativeTotalGuesses != 0 else 0

    cumulativeMechanismRecall = cumulativeMechanismHits/cumulativeMechanismAnswers if cumulativeMechanismAnswers != 0 else 1
    cumulativeEffectRecall = cumulativeEffectHits/cumulativeEffectAnswers if cumulativeEffectAnswers != 0 else 1
    cumulativeAdviseRecall = cumulativeAdviseHits/cumulativeAdviseAnswers if cumulativeAdviseAnswers != 0 else 1
    cumulativeIntRecall = cumulativeIntHits/cumulativeIntAnswers if cumulativeIntAnswers != 0 else 1
    cumulativeTotalRecall = cumulativeTotalHits/cumulativeTotalAnswers if cumulativeTotalAnswers != 0 else 1

    cumulativeMechanismF = (2*cumulativeMechanismPrecision*cumulativeMechanismRecall)/(cumulativeMechanismPrecision + cumulativeMechanismRecall) if (cumulativeMechanismPrecision + cumulativeMechanismRecall) != 0 else 0
    cumulativeEffectF = (2*cumulativeEffectPrecision*cumulativeEffectRecall)/(cumulativeEffectPrecision + cumulativeEffectRecall) if (cumulativeEffectPrecision + cumulativeEffectRecall) != 0 else 0
    cumulativeAdviseF = (2*cumulativeAdvisePrecision*cumulativeAdviseRecall)/(cumulativeAdvisePrecision + cumulativeAdviseRecall) if (cumulativeAdvisePrecision + cumulativeAdviseRecall) != 0 else 0
    cumulativeIntF = (2*cumulativeIntPrecision*cumulativeIntRecall)/(cumulativeIntPrecision + cumulativeIntRecall) if (cumulativeIntPrecision + cumulativeIntRecall) != 0 else 0
    cumulativeTotalF = (2*cumulativeTotalPrecision*cumulativeTotalRecall)/(cumulativeTotalPrecision + cumulativeTotalRecall) if (cumulativeTotalPrecision + cumulativeTotalRecall) != 0 else 0

    print("\tmechP\tmechR\tmechF\teffP\teffR\teffF\tadvP\tadvR\tadvF\tintP\tintR\tintF\ttotP\ttotR\ttotF")
    print(f"S:\t{mechanismPrecision}\t{mechanismRecall}\t{mechanismF}\t{effectPrecision}\t{effectRecall}\t{effectF}\t{advisePrecision}\t{adviseRecall}\t{adviseF}\t{intPrecision}\t{intRecall}\t{intF}\t{totalPrecision}\t{totalRecall}\t{totalF}")
    print(f"C:\t{cumulativeMechanismPrecision}\t{cumulativeMechanismRecall}\t{cumulativeMechanismF}\t{cumulativeEffectPrecision}\t{cumulativeEffectRecall}\t{cumulativeEffectF}\t{cumulativeAdvisePrecision}\t{cumulativeAdviseRecall}\t{cumulativeAdviseF}\t{cumulativeIntPrecision}\t{cumulativeIntRecall}\t{cumulativeIntF}\t{cumulativeTotalPrecision}\t{cumulativeTotalRecall}\t{cumulativeTotalF}")


if __name__ == "__main__":
    evaluate()