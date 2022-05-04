import numpy as np
import pandas as pd
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import csv

output_path = "pred.csv"

# def bag_of_words(data):
#     tk = TweetTokenizer()
#     tokenised = data["text"].apply(tk.tokenize)
#
#     # filter out stopwords and non words - todo: decide on whether we want to filter
#     # todo remove all non words?????
#     allowed_words = set(words.words("en")) - set(stopwords.words("english"))
#
#     stemmer = PorterStemmer()
#     tokenised = np.array([[stemmer.stem(i) for i in x if i in allowed_words] for x in tokenised])
#     # tokenised = np.array([[i for i in x if i in allowed_words] for x in tokenised])
#     return tokenised


def bag_of_words(data):
    text = data["text"]
    vectoriser = CountVectorizer()
    X = vectoriser.fit_transform(text)
    y = data["sentiment"] if "sentiment" in data.columns else None
    return X.toarray(), y

def output_pred_csv(data_x, pred_y):
    header = ["id", "sentiment"]
    with open(output_path, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        data = [[data_x.iloc[i]["id"], pred_y[i]] for i in range(0, len(pred_y))]
        writer.writerows(data)



if __name__ == "__main__":
    data_path = "data/Train.csv"
    test_data_path = "data/Test.csv"

    data = pd.read_csv(data_path)
    test_data = pd.read_csv(test_data_path)



