This system was coded by Jacob Johnson u0403624 for CS6390 at the University of Utah, Spring 2022

For more technical details into how the system works, as well as some evaluation data, see README.pdf

The main scripts are:
    demo.py : the main way to run the code with inputs and get results
    evaluate.py : can run my system on the entire devset or testset and aggregates results
    utils.py : contains the logic of my system, demo.py and evaluate.py both depend on this file
Additionally, I have included some other scripts that made my development easier/possible:
    drugNER.py : this is the code I wrote to train the spaCy NER model for drug entitiy recognition
    readDrugNERdocs.py : I wrote this to extract all the documents and annotations from their xml files
    extractGoldVectors.py : the name is pretty straightforward, this extracts gold context vectors from the training set