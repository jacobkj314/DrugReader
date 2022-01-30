from xml.etree import ElementTree as ET
import spacy
parse = spacy.load("en_core_web_lg")

tree = ET.parse("Abarelix_ddi.xml")
doc = tree.getroot().find("document")
for sentence in doc.findall("sentence"):
    sentenceStr = sentence.text
    sentenceSpacy = parse(sentenceStr)

