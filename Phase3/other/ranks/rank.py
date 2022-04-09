#This is a helper script that I used to rank features and groups of features
#I was going to make a bigger deal about it in my writeup, but I decided it would take too much space to describe, 
#   but I spent a few hours on it, so I felt like it deserved to be part of what I turn in
"""
Example:
Lets say we have a system with N candidate features. An exhaustive ablation test would cover 2^N - 1 systems:
for N = 3:
    A B C
    A B  
    A   C
    A    
      B C
      B  
        C
    (there is no system without at least 1 extracted feature)

Now lets say we rank all the systems by their output evaluation and obtain this ranking:
    1 A B  
    2 A B C
    3 A    
    4 A   C
    5   B  
    6   B C
    7     C
For each position, if a system used a feature, include its ranking, otherwise, include 0:
      A B C
    1 1 1 0
    2 2 2 2
    3 3 0 0
    4 4 0 4
    5 0 5 0
    6 0 6 6
    7 0 0 7
Take the average of each value in a feature's column:
    A: 1.428
    B: 2
    C: 2.714
    (A higher average means that systems with that feature generally had a worse rank)

This can be generalized to sets of features.
Say we want to treat A B as a single feature,
First, get rid of all systems that don't use A and B, or don't not use A and B:
    1 A B   <- OK; uses A and B
    2 A B C <- OK; uses A and B
    X A     <- X; uses A but not B
    X A   C <- X; uses A but not B
    X   B   <- X; uses B but not A
    X   B C <- X; uses B but not A
    3     C ,_ OK; doesn't use A or B
    (In general, grouping K features together will remove 1/(2^(K-1)) of your systems, in this case half of them)
Now just do the same analysis:
    1 AB  
    2 AB C
    3    C

      AB C
    1  1 0
    2  2 2
    3  0 3

    AB: 1
    C: 1.667

This script does the additional step of normalizing this rank onto the range 1 (for the highest possible average rank)
    to -1 (for the lowest possible average rank)


If you want to experiment with it, just run it like
python3 rank.py <features from featureList>
for example
python3 rank.py dep2 dep1 rootDep
"""
from math import log2 as lg
from statistics import mean
import pickle
import sys

featureList = ["path1len", "path1", "dep1", "peak", "dep2", "path2", "path2len", "swapped", "surfdist", "root", "rootPath", "rootDep", "rootLen"]
output = pickle.load(open("ranks", "rb"))

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