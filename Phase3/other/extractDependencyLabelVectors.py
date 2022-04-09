#this is the script that I used to extract the embedding-like vectors to represent dependency labels to my classifier
import pickle
import spacy
from sklearn.linear_model import LogisticRegression
from numpy import concatenate

gold: list[list[tuple[str, list[tuple[int, int, str]], list[tuple[int, int, str]]]]] = pickle.load(open("Train/TRAIN", "rb"))
nlp = spacy.load("en_core_web_lg")

i = 0

data = list()
labels = list()
for doc in gold:
    for sentence, _, _ in doc:
        s = nlp(sentence)
        data.extend([concatenate((w.vector, w.head.vector)) for w in s])
        labels.extend([w.dep_ for w in s])
        print(i)
        i += 1

model = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1)
model.fit(data, labels)

count = len(model.classes_)

vectors = dict()
for i in range(count):
    vectors[model.classes_[i]] = model.coef_[i]
pickle.dump(vectors, open("Phase3/vectors/dependencyVectors", "wb"))