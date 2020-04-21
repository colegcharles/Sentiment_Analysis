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
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


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
            processedTweets.append((self._sanitize(tweet.Tweets), tweet.Label)) # 2 is Label and 4 is tweet
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
    #wordFeatures = wordlist.keys()

    return wordlist


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

    return [[tweet.full_text,None] for tweet in tweets]

#def standardizer(file):

    '''x = data2.loc[:, data2.columns].values

    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['x1', 'x2'])
    finalDf = pd.concat([principalDf, data[['income']]], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    sgd = SGDClassifier(random_state=0, loss='log').fit(X_train, y_train)
    y_pred = sgd.predict(X_test)'''

def getSGD(train, test):
    x = [" ".join(map(str, element[0])) for element in train]
    y = [element[1] for element in train]
    values = array(x)
    label = LabelEncoder()
    encoded = label.fit_transform(values)
    one_hot = OneHotEncoder(sparse=False)
    encoded = encoded.reshape(len(encoded),1)
    xEncoded = one_hot.fit_transform(encoded)
    print(xEncoded)


    df = pd.DataFrame(columns=["Tweets","Label"])



    # gotta encode this
    # dfTrain = pd.DataFrame(train, columns=["Tweets", "Label"])
    # encodeTrain = pd.get_dummies(df, dummy_na=False, drop_first=True)



    '''X_train = [[element[0]] for element in train] # returns a list of tweets stored in the first index from the tuple element
    y_train = [element[1] for element in train] # returns a list of labels stored in the second index

    X_test = [[element[0]] for element in test]
    # y_test = [element[1] for element in test] are all none so does it matter?'''

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    sgd = SGDClassifier(random_state=0, loss='log').fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    print('The accuracy of SGD classifier on test set: {:.2f}'.format(sgd.score(X_test, y_test)))


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
deleteRows = file[file.Label == "neutral"].head(neutralLength- 400)
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
save_dict = open("wordlist.pickle","wb")
pickle.dump(wordList, save_dict, protocol=pickle.HIGHEST_PROTOCOL)
save_dict.close()

trainingFeatures = nltk.classify.apply_features(extractFeatures, sanitizedTrain)
classifier = nltk.NaiveBayesClassifier.train(trainingFeatures)

# sanitize the testing tweets and check accuracy
sanitizedTest = tweetSanitizer.processTweets(dfTest)
testFeatures = nltk.classify.apply_features(extractFeatures, sanitizedTest)

# NBResult = [NBTrain.classify(extractFeatures(tweet[0])) for tweet in sanitizedTest]

# find accuracy: should be like 70%
print("Classifier accuracy percent: %.2f%%" % (nltk.classify.accuracy(classifier, testFeatures) * 100))

# save classifier so we don't train it every time
save_classifier = open("NBayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()




print('nb result is: ' + NBResult)


#print("Overall Positive Sentiment")
'''print("Positive Sentiment Percentage = " + str(100*NBResult.count('positive')/len(NBResult)) + "%")
print("Negative Sentiment Percentage = " + str(100*NBResult.count('negative')/len(NBResult)) + "%")
print("Irrelevant percentage = " + str(100*NBResult.count('irrelevant') / len(NBResult)) + "%")
print("Neutral percentage = " + str(100*NBResult.count('neutral') / len(NBResult)) + "%")
'''
# print("Positive in terms of negative only = " + str(100*NBResult.count('positive')/(NBResult.count('positive') + NBResult.count('negative'))) + "%")
# print("Negative in terms of positive only = " + str(100*NBResult.count('negative')/(NBResult.count('positive') + NBResult.count('negative'))) + "%")

# ----------------------------------GCD---------------------------------------
#getSGD(sanitizedTrain, sanitizedTest)