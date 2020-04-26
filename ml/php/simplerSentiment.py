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


def pickleIt(classifier, name):
    save_classifier = open("Pickles/"+name+".pickle", "wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


pos = open("positive.txt", "r").read()
neg = open("negative.txt", "r").read()

all_words = []
documents = []

# J is adjective. Can add "R" for adverb and "V" for verb
allowedWordTypes = ["J"] # only interested in adjectives

for r in pos.split('\n'):
    documents.append((r, "pos"))
    words = word_tokenize(r)
    partOfSpeech = nltk.pos_tag(words)
    for w in partOfSpeech:
        if w[1][0] in allowedWordTypes:
            all_words.append(w[0].lower())

for r in neg.split('\n'):
    documents.append((r, "neg"))
    words = word_tokenize(r)
    partOfSpeech = nltk.pos_tag(words)
    for w in partOfSpeech:
        if w[1][0] in allowedWordTypes:
            all_words.append(w[0].lower())

# save words
saveDoc = open("Pickles/documents.pickle", "wb")
pickle.dump(documents, saveDoc)
saveDoc.close()

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

saveFeat = open("Pickles/features.pickle", "wb")
pickle.dump(word_features, saveFeat)
saveFeat.close()

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)

saveSets = open("Pickles/featuresets.pickle", "wb")
pickle.dump(featuresets, saveSets)
saveSets.close()

# positive data example:
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

########################################### Regular NB ############################

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

# save it
pickleIt(classifier, "simpleNB")

########################################### Multinomial NB ############################

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# save it
pickleIt(MNB_classifier, "simpleMNB")

########################################### Logistic Regression_classifier ############################
# this gives a convergence warning.
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("Logistic Regression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,
                                                                                 testing_set))*100)
# save it
pickleIt(LogisticRegression_classifier, "simpleLogReg")

########################################### SGD ############################

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGD accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

# save it
pickleIt(SGDClassifier_classifier, "simpleSGD")

########################################### Linear vector classifier ############################

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# save it
pickleIt(LinearSVC_classifier, "simpleLinSVC")

########################################## Classifier we created ###########################################
voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

# Can save it, but this doesn't get trained so no need to
# pickleIt(voted_classifier, "simpleMyClass")
