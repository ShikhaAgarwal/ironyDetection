"""
	This file contains code for Naive Bayes on bag of words as the only feature set.
	It calculates f1_score on 5-fold cross validation.
	The cross val f1-score is 0.66.
	The test f1-score is 0.60
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import *
from sklearn.model_selection import KFold

def load_dataset(filename):
	with open(filename, 'r') as f_in:
		next(f_in)
		column23 = [ cols[1:3] for cols in csv.reader(f_in, delimiter="\t") ]
		Y = [int(col[0]) for col in column23]
		X = [col[1] for col in column23]
		X = np.array(X)
		Y = np.array(Y)
	return X, Y

def init():
	count_vectorizer = CountVectorizer()
	model = MultinomialNB()
	return count_vectorizer, model

def remove_punctuation(my_str):
	punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
	# remove punctuation from the string
	no_punct = ""
	for char in my_str:
	   if char not in punctuations:
	       no_punct = no_punct + char

	return no_punct

def train(count_vectorizer, model, X_train, Y_train):
	word_counts = count_vectorizer.fit_transform(X_train)
	model.fit(word_counts, Y_train)

def predict(count_vectorizer, model, X_test, Y_test):
	word_counts = count_vectorizer.transform(X_test)
	y_pred = model.predict(word_counts)
	return y_pred

# def train(count_vectorizer, model, X_train, Y_train):
# 	X_tr = []
# 	for sentence in X_train:
# 		X_tr.append(remove_punctuation(sentence))

# 	word_counts = count_vectorizer.fit_transform(X_tr)
# 	model.fit(word_counts, Y_train)
# 	# examples = ["Oh such an irony!!", "My name is Shika", "I did nothing this weekend"]
# 	# example_counts = count_vectorizer.transform(examples)
# 	# y_pred = model.predict(example_counts)
# 	# print y_pred

# def predict(count_vectorizer, model, X_test, Y_test):
# 	X_ts = []
# 	for sentence in X_test:
# 		X_ts.append(remove_punctuation(sentence))

# 	word_counts = count_vectorizer.transform(X_ts)
# 	y_pred = model.predict(word_counts)
# 	# for i in range(len(y_pred)):
# 	# 	if y_pred[i] == 0 and Y_test[i] != y_pred[i]:
# 	# 		print X_test[i]
# 	# 		break
# 	# print sum(y_pred == Y_test)
# 	return y_pred

def train_cv(count_vectorizer, model, X, Y):
	k_fold = KFold(n_splits=5, shuffle=True, random_state=3)
	split = k_fold.split(X)
	scores = []
	confusion = np.array([[0, 0], [0, 0]])
	for train_indices, test_indices in split:
	    X_train = X[train_indices]
	    Y_train = Y[train_indices]

	    X_val = X[test_indices]
	    Y_val = Y[test_indices]

	    train(count_vectorizer, model, X_train, Y_train)
	    predictions = predict(count_vectorizer, model, X_val, Y_val)

	    confusion += confusion_matrix(Y_val, predictions)
	    score = f1_score(Y_val, predictions)
	    scores.append(score)
	return confusion, scores


root = "/Users/shikha/UMass/fall2017/NLP/projects/irony/SemEval2018-Task3/datasets"
filename = root + "/train/abc"
X, Y = load_dataset(filename)
train_len = int(0.8 * X.shape[0])
X_train = X[0:train_len]
Y_train = Y[0:train_len]
X_test = X[train_len:]
Y_test = Y[train_len:]

count_vectorizer, model = init()

confusion, scores = train_cv(count_vectorizer, model, X_train, Y_train)
print "Total tweets classified:", X.shape[0]
print "F1 Score:", sum(scores)/len(scores)
print "Confusion matrix:"
print confusion

# Re-train with all data
train(count_vectorizer, model, X_train, Y_train)
y_pred = predict(count_vectorizer, model, X_test, Y_test)
print "F1 score on test data:", f1_score(Y_test, y_pred)
print "Confusion Matrix: "
print confusion_matrix(Y_test, y_pred)
print 'precision', precision_score(Y_test, y_pred)
print 'accuracy', accuracy_score(Y_test, y_pred)

plt.matshow(confusion_matrix(Y_test, y_pred), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()