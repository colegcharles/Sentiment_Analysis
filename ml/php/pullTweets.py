import tweepy as tw
import pandas as pd

'''
DON'T TOUCH
'''
consumer_key = 'MNv539z4C74XY7Ic4nl8dEpfd'
consumer_secret = 'ZJwNtgVEK3o2yy9tEIWmwbMhVkpmtmZtRK4uz3QtvyQNYIbxLF'
access_token = '1223716040548417539-REnwuuX2f2rvbgaqizIiOdpCpqAM6A'
access_token_secret = 'g2sKZzMYUiqAx5zHPWDX9jLO7S889NhtwaxCF51egpD3e'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True) # WILL KEEP US FROM LOSING LICENSE



def get_tweets(trainFile):
    # post a tweet from python
    # api.update_status("Look, I'm tweeting from #Pyhton in my #earthanalytics class!")

    trainingData = []

    for tweet in trainFile.itertuples():
        try:
            # can't user cursor.
            status = api.get_status(tweet[3])

            # append to data
            trainingData.append(status.text)

        except:
            # idk why some tweets can't be pulled up. might be a change in privacy settings.
            # use none as a placeholder then we will remove these tuples
            #trainingData.append(None)

            #just remove tuple now
            trainFile = trainFile.drop(axis=0, index=tweet.Index)
            print("DID NOT WORK")
            continue

    # append tweets at the end
    trainFile["Tweets"] = trainingData

    # create and write to csv file
    trainFile.to_csv("data.csv", index=False,header=True)


def main():
    #how to get tweet by id
    # test = api.get_status(file._get_value(0,2,takeable=True))
    # get a single value of df
    # file.iloc[0][2]

    # read corpus.csv
    file = pd.read_csv("corpus.csv", names=["Topic", "Label", "tID"])
    get_tweets(file)

main()