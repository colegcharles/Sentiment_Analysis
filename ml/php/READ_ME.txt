SanitizeTweets class is inside MasterScript.py. If you call import it from SanitizeTweets.py it expects a list for some reason.
If data.csv is not created, MasterScript.py will call pullTweets.py and create it. This process takes hours.
