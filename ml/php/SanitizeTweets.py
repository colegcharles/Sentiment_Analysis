import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import pandas as pd

'''
    We are sanitizing the data. Removing all punctuation, '#', and any handles. 
    We are also changing the spelling of words, caaaaarrrr is changed to car
    Also, everything is now lowercase to prevent any spelling errors
        cAr != car 
'''


class SanitizeTweets:
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    def processTweets(self, list_of_tweets=pd.DataFrame(columns=["temp"])):
        processedTweets =[]
        for tweet in list_of_tweets.itertuples():
            processedTweets.append((self._sanitize(tweet.Tweets), tweet.Label)) # 2 is Label and 4 is tweet
        return processedTweets

    def _sanitize(self, tweet):
        # here the tweet var we are receiving is already tweet.text
        tweet = tweet.lower()

        # remove URLs, usernames, #, and repeated characters
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # replace it with URL
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # replace it with AT_USER
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # keep word in hashtag but remove #
        tweet = word_tokenize(tweet) # tokenize words

        return [word for word in tweet if word not in self._stopwords]


# trainingData is the corpus.csv tweets
trainingData = []

# testData is the query
testData = []

# might just use the scikit and split up the corpus data
tweetSanitizer = SanitizeTweets()
sanitizedTrain = tweetSanitizer.processTweets(trainingData)
sanitizedTest = tweetSanitizer.processTweets(testData)