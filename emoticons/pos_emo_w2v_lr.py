from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import feature_extractor as fe

def initialize_classifier(w2v):
	clf = Pipeline([

        ('union', FeatureUnion(
        	transformer_list=[
        		('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=fe.LemmaTokenizer())),
        		# ('vectorizer', Pipeline([
        		# 	('count', CountVectorizer(ngram_range=(1, 2))),
        		# 	('tfidf', TfidfTransformer())
        		# ])),
        		('body_w2v', Pipeline([
                    ('pos', fe.TfidfEmbeddingVectorizer(w2v))  # returns a list of dicts
                    # ('vect', DictVectorizer())  # list of dicts -> feature matrix
                ])),
        		('body_pos', Pipeline([
                	('pos', fe.PosTags()),  # returns a list of dicts
                	('vect', DictVectorizer())  # list of dicts -> feature matrix
            	])),
                ('body_stats', Pipeline([
                    ('word_stats', fe.WordFeature()),
                    ('vect', DictVectorizer())  # list of dicts -> feature matrix
                ]))
            ],
            # weight components in FeatureUnion
	        transformer_weights={
    	        'vectorizer': 1.0,
            	'body_stats': 1.0,
                'body_pos': 1.0,
                'body_w2v': 1.0
        	},
        )),

        ('classifier', LR())
    ])
	return clf