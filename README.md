# ConstraintFinder
A neural network that can find sentences containing constraints from txt business document written in natural language.

## Requirements and guide to run
The software is written entirely in Python 3.9.5 so is recommended to use this version to be sure everything will work. Nonetheless, I'm pretty sure that every versione above or equal to 3.9.x is good. To run the code you have to install the following libraries: Gensim, NLTK, TensorFlow, Pickle, NumPy, Matplotlib and nnfs.
This software works as an "input-maker" for the H. Van der Aa software at [this link](https://github.com/hanvanderaa/declareextraction). I made an extension to this program to read sentences from .txt files, you'll find configuration, pom.xml and the Java file in the DeclareExtraction folder. I think you'll be good by simply overwrite the Java file in the other project, but, since I had some troubles, everything you need should be in the folder.
