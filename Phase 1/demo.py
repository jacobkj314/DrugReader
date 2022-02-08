import pickle

train = pickle.load(open("Train/TRAIN", "rb"))

for doc in train:
    for sentence, drugs, interactions in doc:
        if len(interactions) != 0:
            print("\n" + sentence + ":")
        for first, second, _ in interactions:
            print(sentence[drugs[first][0]:drugs[first][1]] + ", " + sentence[drugs[second][0]:drugs[second][1]])
