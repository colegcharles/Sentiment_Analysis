import random
import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


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


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


def loadPickle(path):
    open_file = open(path, "rb")
    obj = pickle.load(open_file)
    open_file.close()
    return obj


documents = loadPickle("Pickles/documents.pickle")
'''documents = pickle.load(documents_f)
documents_f.close()'''

word_features = loadPickle("Pickles/features.pickle")
'''word_features = pickle.load(word_features5k_f)
word_features5k_f.close()
'''

featuresets = loadPickle("Pickles/featuresets.pickle")
'''featuresets = pickle.load(featuresets_f)
featuresets_f.close()'''

random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]

############################################################ Load classifiers ##################################

classifier = loadPickle("Pickles/simpleNBAirline.pickle")
# classifier = loadPickle("Pickles/simpleNB.pickle")

'''classifier = pickle.load(open_file)
open_file.close()
'''

MNB_classifier = loadPickle("Pickles/simpleMNBAirline.pickle")
# MNB_classifier = loadPickle("Pickles/simpleMNB.pickle")
'''MNB_classifier = pickle.load(open_file)
open_file.close()
'''

LogisticRegression_classifier = loadPickle("Pickles/simpleLogRegAirline.pickle")
# LogisticRegression_classifier = loadPickle("Pickles/simpleLogReg.pickle")
'''LogisticRegression_classifier = pickle.load(open_file)
open_file.close()
'''

LinearSVC_classifier = loadPickle("Pickles/simpleLinSVCAirline.pickle")
# LinearSVC_classifier = loadPickle("Pickles/simpleLinSVC.pickle")
'''LinearSVC_classifier = pickle.load(open_file)
open_file.close()'''

SGD_classifier = loadPickle("Pickles/simpleSGDAirline.pickle")
# SGD_classifier = loadPickle("Pickles/simpleSGD.pickle")
'''SGD_classifier = pickle.load(open_file)
open_file.close()'''

SVC_classifier = loadPickle("Pickles/simpleSVCAirline.pickle")
'''SVC_classifier = pickle.load(open_file)'''

BNB_classifier = loadPickle("Pickles/simpleBNBAirline.pickle")
# BNB_classifier = loadPickle("Pickles/simpleBNB.pickle")

# run it through and see which one does best
voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  LogisticRegression_classifier,
                                  SGD_classifier,
                                  SVC_classifier,
                                  BNB_classifier)
