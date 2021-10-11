# ConstraintFinder
A neural network that can find sentences containing constraints from txt formal documents written in natural language.

## Requirements
The software is written entirely in Python 3.9.5 so is recommended to use this version to be sure everything will work. Nonetheless, I'm pretty sure that every 3.9.x version will be good. To run the code you have to install the following libraries: Gensim, NLTK, TensorFlow, Pickle, NumPy, Matplotlib and nnfs.

## Data source
In the data folder you'll find 3 files. The training ones are revised sentences taken from [here](https://www.uniroma1.it/sites/default/files/field_file_allegati/14824_acsai_-_2021-22_-_admission_procedures.pdf) while the test ones are revised sentences taken from [here](http://www.laziodisco.it/wp-content/uploads/2021/06/Call-for-the-Right-to-Education-2021-2022.pdf).

## DeclareExtraction program and how to connect with it
My program works as an "input-maker" for the H. Van der Aa software at [this link](https://github.com/hanvanderaa/declareextraction). 
This software takes an input sentence and returns a Declare constraint, if found. I made an add-on for this program that makes it able to take strings from .txt files and find Declare constraints for every sentence in the file. I'm a little bit rusty in Java so I had some troubles to run the DeclareExtraction project, so in the folder in this repo there are my configurations file to run it with Eclipse (and JRE 1.8) and the new DeclareExtractor.java file (the one with the add-on). Only thing you have to made is download the code from the link above, put the declareextraction-master folder in your ConstraintFinder root folder and copy-paste my DeclareExtraction folder (and overwrite data, obv).

## Sentence format
The Java project expects a .txt file with one line per sentence.

## ConstraintFinder - how to run
If you meet the requirements you can follow this short guideline to run my project:
### File format
The program takes in input two text files to train its neural network and one text file to make predictions. The file can have multiple sentences per line and multiple line per sentence, the only thing you need to check is that every sentence needs to end with a correct punctuation.

### Train
You have to train the model to use it for the first time. There are two different implementations for two types of word embedding (Word2vec, Tensorflow tokenizer):
For Word2vec:

```Python
M, length, wvec = use_w2vec_model("./models/_.model", "./data/try.txt", "./data/try2.txt", training=True)
```

This function returns:
  - M, the model.
  - length, the length of the longest sentence in the training dataset. We need this for predictions (and I'll explain why in a moment).
  - wvec, the Word2vec dictionary.

And takes in input:
  - The path where to save the model.
  - The path of NON-constraint sentences (the sentences that don't contain constraints).
  - The path of constraint sentences.
  - The training parameter (True in this case).

For TF model:

```Python
nM, tokenizer = use_model("./models/_.model", "./data/try.txt", "./data/try2.txt", training=True)
```

This function returns:
  - nM, the model.
  - tokenizer, the Tensorflow tokenizer object (we need it later).
