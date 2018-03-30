import nltk
import numpy as np
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

def tokenize(sentence):
    tokenizer = nltk.TweetTokenizer(strip_handles=True)
    return tokenizer.tokenize(sentence.decode('utf-8'))

class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = nltk.stem.WordNetLemmatizer()
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]

class WordFeature(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        res = []
        file1 = open('complexEmoticons.txt', 'w')
        file2 = open('simpleEmoticons.txt', 'w')
        for sentence in posts:
            feats = {}
            words = tokenize(sentence)
            countExclamation = 0
            emoticon = 0
            i = 0
            length = len(words)
            while i < length:
                w = words[i]
                if (w == ':' or w == '::') and (i+2) < length:
                    x = 0
                    if words[i+2] == ':':
                        x = 3
                    elif words[i+2] == '::':
                        x = 2
                    if x != 0:
                        file1.write(words[i+1].encode('utf-8') + "\n")
                        emoticon += 1
                        i += x
                        continue
                if w != ':' and ':' in w:
                    file2.write(w.encode('utf-8') + "\n")
                    emoticon += 1
                    i += 1
                    continue
                countExclamation += w.count('!')
                i += 1
            # for i, word in enumerate(words):
                # countExclamation += word.count('!')

            feats['length'] = len(sentence)
            feats['exclamation'] = countExclamation
            feats['emoticon'] = emoticon
            res.append(feats)
        file1.close()
        file2.close()
        return res

class PosTags(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
    	res = []
    	for sentence in posts:
    		words = tokenize(sentence)
    		tags = nltk.pos_tag(words)
    		feats = {}
    		countE = 0
    		for w,t in tags:
    			feats[w] = t
    			# countE += w.count('!')
    		# feats['length'] = len(sentence)
    		# feats['exclamation'] = countE
    		res.append(feats)
    	# print "--------------------------------------------------"
    	# print res[0]
    	return res

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])