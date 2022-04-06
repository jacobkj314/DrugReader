#This is a helper script that I used to rank features and groups of features
from math import log2 as lg
from statistics import mean
import pickle
import sys

featureList = ["path1len", "path1", "dep1", "peak", "dep2", "path2", "path2len", "swapped", "surfdist", "root", "rootPath", "rootDep", "rootLen"]
output = pickle.load(open("Phase3/ranks", "rb"))

def minRank(x):
    return sum([i for i in range(1, 2 ** (x-1) + 1)]) / (2 ** x - 1)

def maxRank(x):
    return sum([i for i in range(2 ** (x-1), 2 ** x)]) / (2 ** x - 1)

def rank(ranks):
    m = minRank(int(lg(len(ranks)+1)))
    M = maxRank(int(lg(len(ranks)+1)))
    return (M + m - 2*mean(ranks)) / (M - m)#transform the mean rank to fall between -1 and 1

def getEqual(features: list[bool]):
    first = features[0]
    for feature in features:
        if feature != first:
            return None
    return first

#this takes in an array ranked arrays of features, and an array of which collective features are being evaluated
def project(ranks: list[list[bool]], features: list[bool]):
    result = list()
    for r in ranks:
        r = getEqual([r[i] for i, e in enumerate(features) if e])
        if r is not None:
            result.append(r)
    return result

def number(ranks:list[bool]):
    return [i+1 if e else 0 for i,e in enumerate(ranks)]

if __name__ == "__main__":
    f = [True if f in sys.argv else False for f in featureList]
    print(rank(number(project(output, f))))