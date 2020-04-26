import pandas as pd

file = pd.read_csv("data.csv")

positive = open("positive.txt", "a")

file = file[file.Label != "neutral"]
file = file[file.Label != "irrelevant"]

pos = file[file.Label == "positive"]
neg = file[file.Label == "negative"]

for row in pos.itertuples():
    try:
        positive.write(row.Tweets+"\n")
    except:
        continue

positive.close()


negative = open("negative.txt", "a")
for row in neg.itertuples():
    try:
        negative.write(row.Tweets+"\n")
    except:
        continue

negative.close()