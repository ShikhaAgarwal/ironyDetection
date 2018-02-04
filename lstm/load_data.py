import csv
import numpy as np
import re

def load_dataset(filename):
	print "Loading Dataset..."
	with open(filename, 'r') as f_in:
		next(f_in)
		column23 = [ cols[1:3] for cols in csv.reader(f_in, delimiter="\t") ]
		Y = [int(col[0]) for col in column23]
		X = [col[1] for col in column23]

		X_new = preprocess_data(X)

		X = np.array(X_new)
		Y = np.array(Y)
		print "Done. ", len(X), " tweets loaded!"
	return X, Y

def preprocess_data(X):
	print "removing urls from tweets..."
	X_new = []
	for sentence in X:
		new_sentence = re.sub(r"http\S+", "", sentence)
		X_new.append(new_sentence)
	return X_new

def load_glove_model(gloveFile):
    print "\nLoading Glove Model..."
    f = open(gloveFile,'r')
    wordList = []
    wordVectors = []
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordList.append(word)
        wordVectors.append(embedding)
    wordVectors = np.array(wordVectors)
    print "Done. ",len(wordList)," words loaded!"
    print wordVectors[wordList.index('woman')]

    return wordList, wordVectors
