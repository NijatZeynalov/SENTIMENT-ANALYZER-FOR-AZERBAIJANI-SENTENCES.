#import modules
import pickle
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# load doc into memory
from nltk.corpus import stopwords
import string
import re
from collections import Counter
import codecs


def load_doc(filename):
   # open the file as read only
   file = open(filename, 'r', encoding='utf-8')
   # read all text
   text = file.read()
   # close the file
   file.close()
   return text


# turn a doc into clean tokens
def clean_doc(doc):
   # split into tokens by white space
   tokens = doc.split()
   # prepare regex for char filtering
   re_punc = re.compile('[%s]' % re.escape(string.punctuation))
   # remove punctuation from each word
   tokens = [re_punc.sub('', w) for w in tokens]
   # remove remaining tokens that are not alphabetic
   tokens = [word for word in tokens if word.isalpha()]
   # filter out stop words
   stop_words = set(stopwords.words('azerbaijani'))
   tokens = [w for w in tokens if not w in stop_words]
   # filter out short tokens
   tokens = [word for word in tokens if len(word) > 1]
   return tokens


# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
   # load doc
   doc = load_doc(filename)
   # clean doc
   tokens = clean_doc(doc)
   # update counts
   vocab.update(tokens)


def process_docs(filename, vocab):
   # walk through all files in the folder
   add_doc_to_vocab(filename, vocab)


# save list to file
def save_list(lines, filename):
   # convert lines to a single blob of text
   data = '\n'.join(lines)
   # open file
   file = codecs.open(filename, 'w', encoding='utf8')
   # write text
   file.write(data)
   # close file
   file.close()

# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('negative_words_az.txt', vocab)
process_docs('positive_words_az.txt', vocab)
# print the size of the vocab
print(len(vocab))
# keep tokens with a min occurrence
min_occurane = 1
tokens = [k for k, c in vocab.items() if c >= min_occurane]
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')


def load_doc(filename):
   # open the file as read only
   file = open(filename, 'r', encoding='utf-8')
   # read all text
   text = file.read()
   # close the file
   file.close()
   return text


# turn a doc into clean tokens
def clean_doc(doc):
   # split into tokens by white space
   tokens = doc.split()
   # prepare regex for char filtering
   re_punc = re.compile('[%s]' % re.escape(string.punctuation))
   # remove punctuation from each word
   tokens = [re_punc.sub('', w) for w in tokens]
   # remove remaining tokens that are not alphabetic
   tokens = [word for word in tokens if word.isalpha()]
   # filter out stop words
   stop_words = set(stopwords.words('azerbaijani'))
   tokens = [w for w in tokens if not w in stop_words]
   # filter out short tokens
   tokens = [word for word in tokens if len(word) > 1]
   return tokens


# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
   # load the doc
   doc = load_doc(filename)
   # clean doc
   tokens = clean_doc(doc)
   # filter by vocab
   tokens = [w for w in tokens if w in vocab]
   return ' '.join(tokens)


# load all docs in a directory
def process_docs(filename, vocab):
   lines = list()
   # load and clean the doc
   line = doc_to_line(filename, vocab)
   # add to list
   line = line.split()
   lines.append(line)
   data = lines[0]
   return (data)


def load_clean_dataset(vocab):
   # load documents
   neg = process_docs('negative_words_az.txt', vocab)
   pos = process_docs('positive_words_az.txt', vocab)
   docs = neg + pos
   # prepare labels
   labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
   return docs, labels


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# load all training reviews
docs, labels = load_clean_dataset(vocab)

from sklearn.model_selection import train_test_split


def create_tokenizer(lines):
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts(lines)
   return tokenizer


def define_model(n_words):
   model = Sequential()
   model.add(Dense(100, input_shape=(n_words,), activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(1, activation='sigmoid'))

   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.summary()
   plot_model(model, to_file='output.png', show_shapes=True)
   return model


vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

# create the tokenizer
train_docs, test_docs, y_train, ytest = train_test_split(docs, labels, test_size=0.2)

tokenizer = create_tokenizer(train_docs)
# encode data
Xtrain = tokenizer.texts_to_matrix(train_docs, mode='freq')
Xtest = tokenizer.texts_to_matrix(test_docs, mode='freq')

n_words = Xtest.shape[1]
model = define_model(n_words)
model.fit(Xtrain, y_train, epochs=20, verbose=2)
# evaluate#
loss, acc = model.evaluate(Xtest, ytest, verbose=2)
model.save('sent_azerbaijani.h5')