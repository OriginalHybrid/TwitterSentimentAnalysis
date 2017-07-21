import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize

# ----------------------------------------------------------------------------------------------------------------------



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# ----------------------------------------------------------------------------------------------------------------------


save_docs_f=open("documents.pickle","rb")
documents = pickle.load(save_docs_f)
save_docs_f.close()

# ----------------------------------------------------------------------------------------------------------------------

word_features_f=open("word_features.pickle","rb")
word_features = pickle.load(word_features_f)
word_features_f.close()


# ----------------------------------------------------------------------------------------------------------------------

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets_f=open("feature_sets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()



random.shuffle(featuresets)

# ----------------------------------------------------------------------------------------------------------------------


training_sets = featuresets[:10000]
testing_sets = featuresets[10000:]


# ----------------------------------------------------------------------------------------------------------------------


classifer_f = open("naivebayes.pickle", "rb")
classifer = pickle.load(classifer_f)
classifer_f.close()


# ----------------------------------------------------------------------------------------------------------------------


open_file = open("MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


# ----------------------------------------------------------------------------------------------------------------------


open_file = open("BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


# ----------------------------------------------------------------------------------------------------------------------


open_file = open("LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

# ----------------------------------------------------------------------------------------------------------------------


open_file = open("SGDClassifier_classifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

# ----------------------------------------------------------------------------------------------------------------------


open_file = open("SVC_classifier.pickle", "rb")
SVC_classifier = pickle.load(open_file)
open_file.close()


# ----------------------------------------------------------------------------------------------------------------------


open_file = open("LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

# ----------------------------------------------------------------------------------------------------------------------


open_file = open("NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()


# ----------------------------------------------------------------------------------------------------------------------


voted_classifier = VoteClassifier(classifer,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  SVC_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)
print(" Voted Classifier Algo Accuracy", (nltk.classify.accuracy(voted_classifier,testing_sets))*100)


def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats), (voted_classifier.confidence(feats)*100)

