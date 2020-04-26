import re
from nltk.corpus import stopwords
import SentimentAnalysis as s
import tweepy as tw
from string import punctuation
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
import sys

# pulls keys from twitterStuff
from twitterStuff import *

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)  # WILL KEEP US FROM LOSING LICENSE


def get_tweets(search_words, count=10):
    # collect tweets
    tweets = tw.Cursor(api.search,
                       q=search_words,
                       lang="en",
                       tweet_mode='extended').items(count)

    return [tweet.full_text for tweet in tweets]


class SanitizeTweetText:
    def __init__(self):
        # I don't know why but eâ€ shows up a lot
        self._stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL', 'rt', 'eâ€', '€','â'])

    def processTweets(self, tweets):
        processedTweets = []
        for tweet in tweets:
            processedTweets.append(self._sanitize(tweet))
        return processedTweets

    def _sanitize(self, tweet):
        tweet = tweet.lower()
        # print(tweet)
        # remove URLs, usernames, #, and repeated characters
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # replace it with URL
        # print(tweet)
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # replace it with AT_USER
        # print(tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # keep word in hashtag but remove #
        # print(tweet)
        # sentiment analysis module tokenizes there
        # tweet = word_tokenize(tweet)  # tokenize words

        # so split it?????
        temp = []
        for word in tweet.split():
            if word not in self._stopwords:
                # print(word)
                temp.append(word)
        tweet = " ".join(temp)
        # lWords = [word for word in tweet if word not in self._stopwords]
        return tweet


def animate(labels, posINCREMENT, negINCREMENT):
    # we're going to use the positive and negative labels received from the classifier
    xar = []
    yar = []

    x = 0
    y = 0
    for label in labels:
        x += 1
        # npl dates if you wanna make it date graph
        if label == "pos":
            y += posINCREMENT
        elif label == 'neg':
            # negative bias? make a negative count as half
            y -= negINCREMENT
        xar.append(x)
        yar.append(y)

    return xar, yar

def main():
    # search api and get a list of tweets back
    query = sys.argv[1]
    tweets = get_tweets(query, 100)
    sanitizer = SanitizeTweetText()
    sanitizedTweets = sanitizer.processTweets(tweets)
    labels = []
    conf = []
    # biased towards negative. Make negative counts less
    negINCREMENT = 1  # CHANGE TO 0.5 or 0.3 to combat bias
    posINCREMENT = 1
    negCount = 0
    posCount = 0

    for tweet in sanitizedTweets:
        try:
            sentiment = s.sentiment(tweet)
            conf.append(sentiment[1])
            # print(tweet)
            # print(sentiment)
            # adds the sentiment, pos, neg, to labels
            if sentiment[1] * 100 >= 80:
                labels.append(sentiment[0])
                if sentiment[0] == "neg":
                    negCount += negINCREMENT
                else:
                    posCount += posINCREMENT
            else:
                labels.append("neutral")
                # print("Sentiment confidence too low. Will not be counted.")

        except:
            # print("\n\nCANNOT EVALUATE THE TWEET:")
            # print(tweet+"\n\n")
            labels.append("irrelevant")
            conf.append(None)

    # make a string and test sentiment
    '''
    print(s.sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!")) # positive
    print(s.sentiment("This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horible movie 0/10!")) # negative
    '''

    # make it a dataframe
    # labels, tweets, sanitized tweets
    dictionary = {"Label": labels, "Confidence": conf, "Full Tweet": tweets, "Sanitized Tweet": sanitizedTweets}
    pdData = pd.DataFrame(dictionary)
    pdData.to_csv(r'Data/' + query + 'Data.csv', index=False)

    # now plot it
    style.use("ggplot")
    xar, yar = animate(labels, posINCREMENT, negINCREMENT)
    plt.plot(xar, yar)
    plt.ylabel('Sentiment')
    plt.xlabel('Per tweet')
    plt.title(query + " Sentiment Analysis")
    plt.savefig("graphs/" + query + "LineGraph.png")
    #plt.show()
    print('plot saved')

    # pie chart
    pieLabels = ["Positive", "Negative"]
    posPercent = posCount / (negCount + posCount)
    negPercent = negCount / (negCount + posCount)
    sizes = [posPercent, negPercent]

    explode = (0, 0.1) if posPercent > negPercent else (0.1, 0)  # explode negative

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=pieLabels, autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig("graphs/" + query + "BarGraph.png")
    #plt.show()
    print('figure saved')

main()