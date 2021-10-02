import re
import nltk
import os
from nltk.stem import WordNetLemmatizer
from random import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# retrieve sentences form txt files
# it makes use of nltk library for sentence tokenization
def get_sentences_from_txt(txt_path, testing=False):
    with open(txt_path, "r") as file:
        data = file.read().replace("\n", " ")
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = (sent_tokenizer.tokenize(data.strip()))
    if testing: shuffle(sentences)
    return sentences


def rm_punctuation(sent): return re.sub(r'[^\w\s]', '', sent)


# function that reduces sentences (every verb "reduced" to its base form)
# it makes use of nltk WordNetLemmatizer
def sentence_root(X):
    sentences = X.copy()
    sentences = [rm_punctuation(sent) for sent in sentences]
    wordsplitted = [s.split(" ") for s in sentences]

    stemmer = WordNetLemmatizer()

    for i in range(len(wordsplitted)):
        for j in range(len(wordsplitted[i])):
            wordsplitted[i][j] = stemmer.lemmatize(wordsplitted[i][j], 'v')

    sentences = [" ".join(s) for s in wordsplitted]
    return sentences


# maps sentences to its
def class_mapping(non_activity, activity):
    # creates two lists in the format that follows:
    # [(sentence, class), ...]
    for i in range(len(non_activity)): non_activity[i] = (non_activity[i], 1)
    for i in range(len(activity)): activity[i] = (activity[i], 0)

    # merging the lists and shuffle elements
    sentences = activity + non_activity
    shuffle(sentences)

    X, y = [], []

    # separation of sentences and their classification
    for sentence in sentences:
        X.append(sentence[0])
        y.append(sentence[1])

    # sentences reduction
    sentences = sentence_root(X)

    # returns reducted sentences, class targets and original sentences
    return sentences, y, X


# performs word embedding mapping every word
# to an integer, increasing values
# "I love you, you love me too" will be mapped as follows:
# I -> 0, love -> 1, you -> 2, me -> 3, too -> 4
# the function also normalizes data between -1 and 1
def TF_tofloat_tokenizer(sentences):
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    # word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)
    sequences, max_value = normalizator(sequences.copy())
    # performs word embedding and sets the maximum length of a sequence to 45
    # fills the padding values with -1e-7
    padded = pad_sequences(sequences, dtype="float32", maxlen=45, value=1e-7)

    # max_len = max([len(array) for array in padded])
    #
    # #
    # if len(padded[1]) < max_len:
    #     zeros = np.zeros((max_len - len(padded[1]),))
    #     for i in range(len(padded)): padded[i] = np.concatenate([zeros, padded[i]])

    return padded, max_value, tokenizer


def normalizator(sequences):
    max_value = np.max(np.max(sequences))
    normalizer = max_value / 2

    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            sequences[i][j] -= normalizer
            sequences[i][j] /= normalizer

    return sequences, max_value


# peforms every operation needed for input preparation
# for a normal (non Word2vec) model
def data_preparation(nas_path, as_path):

    non_activity = get_sentences_from_txt(nas_path)
    activity = get_sentences_from_txt(as_path)

    sentences, y, originals = class_mapping(non_activity, activity)

    padded, max_value, tokenizer = TF_tofloat_tokenizer(sentences)

    # mapping = map_sentence_to_PS(sentences, padded)

    return padded, y, originals, max_value, tokenizer


def write_to_file(sentences, w2vec=False):
    path = "./declareextraction-master/DeclareExtraction/nn_outputs"

    if not os.path.exists(path): os.makedirs(path)

    marker = "/w2vec_" if w2vec else "/"

    with open(path + marker + "constraint_sentences.txt", "w") as txt_file:
        for sent in sentences: txt_file.write(sent+"\n")