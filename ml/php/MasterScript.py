'''
1- Build a vocabulary (list of words) of all the words resident in our training data set.
2- Match tweet content against our vocabulary — word-by-word.
3- Build our word feature vector.
4- Plug our feature vector into the Naive Bayes Classifier.
'''
from textblob import TextBlob
import nltk
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
import tweepy as tw
import os
import os.path
from os import path
import pickle
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    # pass a list of classifiers to VoteClassifier
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        # c is classifier, but cannot use classifier name
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)
        return mode(votes)

'''    def confidence(self, features):
        votes = []
        # c is classifier, but cannot use classifier name
        for c in self._classifiers:
            vote = c.classify(features)
            votes.append(vote)

        # counts occurences of most popular votes
        choice = votes.count(mode(votes))
        # does not return % keeps it in decimal
        confidence = choice / len(votes)
        return confidence
'''

'''
DON'T TOUCH
'''
# pulls keys from twitterStuff
from twitterStuff import *
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)  # WILL KEEP US FROM LOSING LICENSE


class SanitizeTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def processTweets(self, list_of_tweets=pd.DataFrame(columns=["temp"])):
        processedTweets = []
        for tweet in list_of_tweets.itertuples():
            processedTweets.append((self._sanitize(tweet.Tweets), tweet.Label))  # 2 is Label and 4 is tweet
        return processedTweets

    def _sanitize(self, tweet):
        # here the tweet var we are receiving is already tweet.text
        tweet = tweet.lower()

        # remove URLs, usernames, #, and repeated characters
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # replace it with URL
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # replace it with AT_USER
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # keep word in hashtag but remove #
        tweet = word_tokenize(tweet)  # tokenize words

        return [word for word in tweet if word not in self._stopwords]


def buildVocab(sanitizedTrainingData):
    # a list of all the words used in the training set
    allWords = []

    for (words, sentiment) in sanitizedTrainingData:
        allWords.extend(words)

    # break it up into a list of distinct words with a frequency as a key
    wordlist = nltk.FreqDist(allWords)
    # wordFeatures = wordlist.keys()

    return wordlist


def extractFeatures(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in wordFeatures:
        features['contains(%s)' % word] = (word in tweet_words)

    return features


def get_tweets(search_words, count=10):
    # collect tweets
    tweets = tw.Cursor(api.search,
                       q=search_words,
                       lang="en",
                       tweet_mode='extended').items(count)

    return [[tweet.full_text, None] for tweet in tweets]


def split(file):
    # shuffles df
    file = file.sample(frac=1)
    length = len(file) // 2
    train = file[:length]
    test = file[length:]
    return train, test


# read corpus.csv
try:
    file = pd.read_csv("data.csv")
except:
    # pullTweets.py will create the data.csv so call it here
    print("Get ready to wait for hours")
    os.system('python pullTweets.py')
    file = pd.read_csv("data.csv")

# remove irrelevant label. not needed since we only search the twitter api for english
file = file[file.Label != "irrelevant"]

# even out the neutral, positive, negative rows (400 neutral rows
neutralLength = len(file[file.Label == "neutral"])
deleteRows = file[file.Label == "neutral"].head(neutralLength - 400)
file = file.drop(deleteRows.index)

# splits the data down the middle
dfTrain, dfTest = split(file)

# sanitize the tweets and train the NB
tweetSanitizer = SanitizeTweets()
sanitizedTrain = tweetSanitizer.processTweets(dfTrain)

# only pass back word list bc we can't pickle dict keys
wordList = buildVocab(sanitizedTrain)
wordFeatures = wordList.keys()

# save word list and vocab created. don't wanna redo it every time
save_dict = open("wordlist.pickle", "wb")
pickle.dump(wordList, save_dict, protocol=pickle.HIGHEST_PROTOCOL)
save_dict.close()

# with above code, training more classifiers is a breeze


############################################################# Regular naive bays ################################

trainingFeatures = nltk.classify.apply_features(extractFeatures, sanitizedTrain)
classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

# sanitize the testing tweets and check accuracy
sanitizedTest = tweetSanitizer.processTweets(dfTest)
testFeatures = nltk.classify.apply_features(extractFeatures, sanitizedTest)

# find accuracy: should be like 70%
print("Classifier accuracy percent: %.2f%%" % (nltk.classify.accuracy(classifier, testFeatures) * 100))

# save classifier so we don't train it every time
save_classifier = open("NBayes.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

############################################################# Multinomial naive bayes ################################
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

MNBclassifier = SklearnClassifier(MultinomialNB())
MNBclassifier.train(trainingFeatures)

# should be about 79.07%
print("Multinomial Naive Bayes accuracy percent: %.2f%%" % (nltk.classify.accuracy(MNBclassifier, testFeatures) * 100))

############################################################# Logistic Regression ################################
from sklearn.linear_model import LogisticRegression, SGDClassifier

# idk if I want to do linear vector classifier
# from sklearn.svm import SVC, LinearSVC, NuSVC

logRegclassifier = SklearnClassifier(LogisticRegression())
logRegclassifier.train(trainingFeatures)

# 82.13% accuracy
print("Logistic regression accuracy percent: %.2f%%" % (nltk.classify.accuracy(logRegclassifier, testFeatures) * 100))

SGDclassifier = SklearnClassifier(SGDClassifier())
SGDclassifier.train(trainingFeatures)

# 78.58% maybe if we use parfit and other stuff to make it work better?
print("SGD accuracy percent: %.2f%%" % (nltk.classify.accuracy(SGDclassifier, testFeatures) * 100))

############################################################# Combine algo ################################
# now combine algorithms to see which of the 4 work the best
''' 
    NOT CURRENTLY WORKING. 
    I think its because we use positive, negative, and neutral (3 labels)
    instead of 2 labels
'''
# we are building a classifier that will be a compilation of all
# mode gives us a way of finding the popular vote
voteClass = VoteClassifier(classifier,
                           MNBclassifier,
                           logRegclassifier,
                           SGDclassifier)
# print("voteClass accuracy percent: %.2f%%" % (nltk.classify.accuracy(voteClass, testFeatures) * 100))

# print("Classification: ", voteClass.classify(testFeatures), "Confidence %: ", voteClass.confidence(testFeatures))
