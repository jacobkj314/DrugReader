import utils

dev = utils.getGold("Dev")

def demo(threshhold: float, vecType: str):
    utils.thr = threshhold
    utils.vectorType = vecType
    utils.removedByAnova = 0

    answers = 0
    guesses = 0
    hits = 0

    for i, doc in enumerate(dev):
        print(i)
        docText = " ".join([sentenceText for sentenceText, _, _ in doc])
        actualRelations = set()
        for sentenceText, drugChars, interactions in doc:
            drugs = [sentenceText[start:end] for start, end, _ in drugChars]
            interactions = [(drugs[first], drugs[second], label) for first, second, label in interactions]
            for interaction in interactions:
                actualRelations.add(interaction)
        #notice that for evaluation, we use the gold entities instead of spacy-extracted entities.
        #use this one #extractedRelations = {(first, second, label) for first, second, label, _ in utils.extractRelationsFromGoldEntities(doc)}
        #extractedRelations = {(first, second, label) for first, second, label, _ in utils.extractRelations(docText)}

        utils.vectorType = "peak-weighted"
        peak = {(first, second, label) for first, second, label, _ in utils.extractRelationsFromGoldEntities(doc)}
        utils.vectorType = "end-weighted"
        end = {(first, second, label) for first, second, label, _ in utils.extractRelationsFromGoldEntities(doc)}
        utils.vectorType = "uniform-weighted"
        uniform = {(first, second, label) for first, second, label, _ in utils.extractRelationsFromGoldEntities(doc)}

        extractedRelations = peak.intersection(end).union(end.intersection(uniform)).union(uniform.intersection(peak))

        print(extractedRelations, "|", actualRelations)

        answers += len(actualRelations)
        guesses += len(extractedRelations)
        hits += len(actualRelations.intersection(extractedRelations))

    precision = hits/guesses if guesses != 0 else 0
    recall = hits/answers if answers != 0 else 1
    fScore = (2*precision*recall)/(precision + recall) if (precision + recall) != 0 else 0

    file = open("results.csv", "a")
    file.write(f"Bagged{threshhold}, {answers}, {guesses}, {hits}, {precision}, {recall}, {fScore}\n")
    file.close()

if __name__ == "__main__":
    for vType in ["peak-weighted", "end-weighted", "uniform-weighted"]:
        i = .55
        while i < 1.00:
            demo(i, vType)
            i += 1



