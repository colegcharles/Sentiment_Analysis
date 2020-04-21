import nltk
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd
import tweepy as tw
import os
import pickle
import sys

'''
DON'T TOUCH
'''
consumer_key = 'MNv539z4C74XY7Ic4nl8dEpfd'
consumer_secret = 'ZJwNtgVEK3o2yy9tEIWmwbMhVkpmtmZtRK4uz3QtvyQNYIbxLF'
access_token = '1223716040548417539-REnwuuX2f2rvbgaqizIiOdpCpqAM6A'
access_token_secret = 'g2sKZzMYUiqAx5zHPWDX9jLO7S889NhtwaxCF51egpD3e'

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


def extractFeatures(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in wordFeatures:
        features['contains(%s)' % word] = (word in tweet_words)

    return features


def get_tweets(search_words, count=10):
    # post a tweet from python
    # api.update_status("Look, I'm tweeting from #Pyhton in my #earthanalytics class!")

    # collect tweets
    tweets = tw.Cursor(api.search,
                       q=search_words,
                       lang="en",
                       tweet_mode='extended').items(count)

    return [[tweet.full_text, None] for tweet in tweets]


query = sys.argv[1]
count = int(sys.argv[2])
testData = get_tweets(query, count)  # returns a list of text
# check tweets
#for tweet in testData:
    #print(tweet, "\n")
dfTest = pd.DataFrame(testData, columns=['Tweets', "Label"])  # sanitize uses pandas
tweetSanitizer = SanitizeTweets()
sanitizedTest = tweetSanitizer.processTweets(dfTest)

classifier_f = open("NBayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

wordlist_f = open("wordlist.pickle", "rb")
wordlist = pickle.load(wordlist_f)
wordlist_f.close()

wordFeatures = wordlist.keys()

NBResult = [classifier.classify(extractFeatures(tweet[0])) for tweet in sanitizedTest]

print("<h1 style='margin-top=5%'>Overall Positive Sentiment</h1><br>")
print("<li class='list-group-item'>Positive Sentiment Percentage = " + str(100 * NBResult.count('positive') / len(NBResult)) + "%</li><br>")
print("<li class='list-group-item'>Negative Sentiment Percentage = " + str(100 * NBResult.count('negative') / len(NBResult)) + "%</li><br>")
print("<li class='list-group-item'>Irrelevant percentage = " + str(100 * NBResult.count('irrelevant') / len(NBResult)) + "%</li><br>")
print("<li class='list-group-item'>Neutral percentage = " + str(100 * NBResult.count('neutral') / len(NBResult)) + "%</li><br>")
