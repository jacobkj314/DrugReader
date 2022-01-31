import os
from xml.etree import ElementTree as ET

wordCount = dict()
def increment(i : int) -> None:
    if wordCount.get(i) == None:
        wordCount[i] = 0
    wordCount[i] += 1

docFolder = "../Train"
for file in os.listdir(docFolder):
    file = os.path.join(docFolder, file)
    doc = ET.parse(file)
    root = doc.getroot()
    for drug in root.iter("entity"):
        #print(drug.get("text"))
        increment(len(drug.get("text").split()))

print(wordCount)

"""
12523 one-word drugs
1747 two-word drugs
371 three-word drugs
106 four-word drugs
15 five-word drugs
2 six-word drugs
1 seven-word drug
Conclusion: start by ignoring BOI tags
"""