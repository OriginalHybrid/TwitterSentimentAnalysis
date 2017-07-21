import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize



print("Hello")
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

print("Hello")

pos_res = open("positive.txt", "r").read()
neg_res = open("negative.txt", "r").read()

documents = []
all_words = []

# ----------------------------------------------------------------------------------------------
#  j is adjective , r is adverb, and v is verb
#  allowed_word_types = ["J","R","V"]

allowed_word_types = ["J","R","V"]

for r in pos_res.split('\n'):
    documents.append((r, "pos"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in neg_res.split('\n'):
    documents.append((r, "neg"))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

print("Hello")

save_docs = open("documents.pickle","wb")
pickle._dump(documents,save_docs)
save_docs.close()

# ------------------------------------------------------------------------------------------
#
# short_pos_words = word_tokenize(pos_res)
# short_neg_words = word_tokenize(neg_res)
#
# for w in short_pos_words:
#     all_words.append(w.lower())
#
# for w in short_neg_words:
#     all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

print("Hello")

# --------------------------------------------------------------------------------------------

word_features = list(all_words.keys())[:5000]


save_features = open("word_features.pickle","wb")
pickle._dump(word_features,save_features)
save_features.close()

# ----------------------------------------------------------------------------------------------

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


#print(find_features(movie_reviews.words("neg/cv000_29416.txt")))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

f_sets = open("feature_sets.pickle","wb")
pickle.dump(featuresets, f_sets)
f_sets.close()

print("Hello")

# -------------------------------------------------------------------------------------------

# #Positive_Data
# training_sets = feature_sets[:1900]
# testing_sets = feature_sets[1900:]


#Negative_Data
training_sets = featuresets[:10000]
testing_sets = featuresets[10000:]

print("Hello")

# ---------------------------------------------------------------------------------------------


classifier = nltk.NaiveBayesClassifier.train(training_sets)

# classifer_f=open("naivebayes.pickle","rb")
# classifer=pickle.load(classifer_f)
# classifer_f.close()

print("Hello")

# ---------------------------------------------------------------------------------------------


print(" Original Naive Bayes Algo Accuracy", (nltk.classify.accuracy(classifier, testing_sets))*100)
classifier.show_most_informative_features(15)

save_classifier = open("OriginalNBC.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_sets)
print(" MNB Classifier Algo Accuracy",(nltk.classify.accuracy(MNB_classifier, testing_sets))*100)

save_classifier = open("MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------

##
### GaussianNB_classifier = SklearnClassifier(GaussianNB())
### GaussianNB_classifier.train(training_sets)
### print(" GaussianNB Classifier Algo Accuracy",(nltk.classify.accuracy(GaussianNB_classifier,training_sets))*100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_sets)
print(" BernoulliNB Classifier Algo Accuracy",(nltk.classify.accuracy(BernoulliNB_classifier, testing_sets))*100)

save_classifier = open("BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_sets)
print(" LogisticRegression Classifier Algo Accuracy",(nltk.classify.accuracy(LogisticRegression_classifier,testing_sets))*100)

save_classifier = open("LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------


SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_sets)
print(" SGDClassifier Classifier Algo Accuracy",(nltk.classify.accuracy(SGDClassifier_classifier,testing_sets))*100)

save_classifier = open("SGDClassifier_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_sets)
print(" SVC Classifier Algo Accuracy",(nltk.classify.accuracy(SVC_classifier,testing_sets))*100)

save_classifier = open("SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_sets)
print(" LinearSVC Classifier Algo Accuracy",(nltk.classify.accuracy(LinearSVC_classifier,testing_sets))*100)

save_classifier = open("LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_sets)
print(" NuSVC Classifier Algo Accuracy",(nltk.classify.accuracy(NuSVC_classifier,testing_sets))*100)

save_classifier = open("NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier,save_classifier)
save_classifier.close()

# ---------------------------------------------------------------------------------------------


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  SVC_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)
print(" Voted Classifier Algo Accuracy",(nltk.classify.accuracy(voted_classifier,testing_sets))*100)


def sentiment(text):
    feats = find_features(text)

    voted_classifier.classify(feats)