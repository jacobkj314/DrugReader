import utils

dev = utils.getGold("Dev")

def demo(threshhold: float):
    utils.thr = threshhold

    answers = 0
    guesses = 0
    hits = 0

    for doc in dev:
        docText = " ".join([sentenceText for sentenceText, _, _ in doc])
        actualRelations = set()
        for sentenceText, drugs, interactions in doc:
            drugs = [sentenceText[start:end] for start, end, _ in drugs]
            interactions = [(drugs[first], drugs[second], label) for first, second, label in interactions]
            for interaction in interactions:
                actualRelations.add(interaction)
            #for first, second, _ in interactions:
                #actualRelations.add((first, second))
        extractedRelations = {(first, second, label) for first, second, label, _ in utils.extractRelations(docText)}
        #extractedRelations = {(first, second) for first, second, _, _ in utils.extractRelations(docText)}

        answers += len(actualRelations)
        guesses += len(extractedRelations)
        hits += len(actualRelations.intersection(extractedRelations))

    print(f"{threshhold}, {answers}, {guesses}, {hits}")


if __name__ == "__main__":
    i = .55
    while i < 1.00:
        demo(i)
        i += .01


