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
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)


documents_f = open("Pickles/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("Pickles/features.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()





featuresets_f = open("Pickles/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

testing_set = featuresets[10000:]
training_set = featuresets[:10000]


############################################################ Load classifiers ##################################

open_file = open("Pickles/simpleNB.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("Pickles/simpleMNB.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("Pickles/simpleLogReg.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("Pickles/simpleLinSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("Pickles/simpleSGD.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

# run it through and see which one does best
voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  LogisticRegression_classifier)




