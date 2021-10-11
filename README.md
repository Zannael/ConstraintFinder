# ConstraintFinder
A neural network that can find sentences containing constraints from txt formal documents written in natural language.

## Requirements
The software is written entirely in Python 3.9.5 so is recommended to use this version to be sure everything will work. Nonetheless, I'm pretty sure that every 3.9.x version will be good. To run the code you have to install the following libraries: Gensim, NLTK, TensorFlow, Pickle, NumPy, Matplotlib and nnfs.

## Data source
In the data folder you'll find 3 files. The training ones are revised sentences taken from [here](https://www.uniroma1.it/sites/default/files/field_file_allegati/14824_acsai_-_2021-22_-_admission_procedures.pdf) while the test ones are revised sentences taken from [here](http://www.laziodisco.it/wp-content/uploads/2021/06/Call-for-the-Right-to-Education-2021-2022.pdf).

## DeclareExtraction program and how to connect with it
My program works as an "input-maker" for the H. Van der Aa software at [this link](https://github.com/hanvanderaa/declareextraction). 
This software takes an input sentence and returns a Declare constraint, if found. I made an add-on for this program that makes it able to take strings from .txt files and find Declare constraints for every sentence in the file. I'm a little bit rusty in Java and I had some troubles to run the DeclareExtraction project, for this reason in the DeclareExtraction folder of this repo there are my configurations file to run it with Eclipse (and JRE 1.8) and the new DeclareExtractor.java file (the one with the add-on). Only thing you have to made is download the code from the link above, put the declareextraction-master folder in your ConstraintFinder root folder and copy-paste my DeclareExtraction folder (and overwrite data, obv).

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

And takes in input:
  - The path where to save the model.
  - The path of NON-constraint sentences (the sentences that don't contain constraints).
  - The path of constraint sentences.
  - The training parameter (True in this case).

### Make predictions
We can use the same functions with different arguments.
For Word2vec:

```Python
M, res = use_w2vec_model("./models/bestw2vec.model", "", "", max_len=length, training=False, doc_path="data/final_test.txt", w2vec_model=wvec)
```

The function returns:
  - M, the model.
  - res, array with (sentence, class_number) elements where class_number is 0: constraint, 1: non constraint

And takes in input:
  - The path from where to load the model.
  - Empty strings for the 2 training paths (yeah ik, awful).
  - max_len=length, the one returned from the training. 
  - The training parameter (False in this case).
  - doc_path is the path from where the model reads the sentences on which it makes predictions. The format of this document is the same as described before.
  - w2vec_model=wvec, the object returned from the training.

**If you want to make predictions without training before, using a pretrained model, make sure you know exactly the length value and you have stored somewhere the wvec object (e.g. with pickle).**

For TF model:

```Python
nM, res = use_model("./models/_.model", "", "", training=False, doc_path="data/final_test.txt", tokenizer=tokenizer)
```

The function returns:
  - nM, the model.
  - res, array with (sentence, class_number) elements where class_number is 0: constraint, 1: non constraint

And takes in input:
  - The path from where to load the model.
  - Empty strings for the 2 training paths.
  - The training parameter (False in this case).
  - doc_path is the path from where the model reads the sentences on which it makes predictions. The format of this document is the same as described before.
  - tokenizer=tokenizer, the object returned from the training.

### Show predictions and save to file
To show the sentences with constraints you can just run the first line of code and print. The second line writes the sentences in a file at declareextraction-master/DeclareExtraction/nn_outputs/. The file is already formatted as the DeclareExtraction program requires.

```Python
cs = get_constraint_sentences(res)
write_to_file(cs, w2vec=False)
```

res is just the array returned with the predictions and w2vec True/False only changes the name of the file created.

### Run DeclareExtraction
If you followed the first part of the README you already have [this](https://github.com/hanvanderaa/declareextraction) program downloaded and placed in your project root folder. You've already copy-pasted my add-on files. Now you can simply run the Java program. If you have your predicted sentences stored somewhere that's not declareextraction-master/DeclareExtraction/nn_outputs/ you have to open the new DeclareExtractor.java file and change this path with yours.

## ConstraintFinder show results
To plot training accuracy, recall and loss just use:

```Python
M.plot(w2vec=False)
```

Where M is a Model object (returned for example after training). This function **saves** the plots in a folder. w2vec parameter is just for the name of the files saved.

If you use: 

```Python
M.plot(show=True)
```

Your IDE will (hopefully 😬) show you the plots (and save 'em as well).

To get validation accuracy, recall and loss:

```Python
import statistics

tup = round(M.validation_accuracies[-1], 3), M.validation_recalls[-1], round(M.validation_losses[-1], 3)
hm = round(statistics.harmonic_mean((tup[0], tup[1])), 3)
print("Validation  & " + str(tup[0]) + " & " + str(tup[1]) + " & " + str(tup[2]) + " & " + str(hm))
```
