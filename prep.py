import numpy as np
import pandas as pd
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import csv
from sklearn.feature_selection import SelectKBest, chi2

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

# report can talk about: ngrams, data cleaning, select k best
class BagOfWords():

    def __init__(self, k=None):
        self.vectoriser = CountVectorizer(max_features=2000)
        self.k = k

    def train_prep(self, data):
        text = data["text"]
        y = data["sentiment"].to_numpy()
        # clean text
        data["text"] = self.clean(data["text"])
        X = self.vectoriser.fit_transform(text)

        if self.k is not None:
            k_best = SelectKBest(chi2, k=self.k)
            k_best.fit(X.toarray(), y)
            indicies = k_best.get_support(indices=True)
            vocabulary = [self.vectoriser.get_feature_names()[i] for i in indicies]
            self.vectoriser.vocabulary = vocabulary
            X = self.vectoriser.transform(text)

        return X.toarray(), y

    def pred_prep(self, data):
        text = data["text"]
        data["text"] = self.clean(data["text"])
        return self.vectoriser.transform(text)

    def clean(self, data_text):
        lowered = data_text.str.strip().str.lower()

        # todo: make optional through parameter
        stemmer = PorterStemmer()
        tk = TweetTokenizer()
        allowed_words = set(words.words("en")) - set(stopwords.words("english"))
        lowered.apply(lambda x: " ".join([stemmer.stem(i) for i in tk.tokenize(x) if i in allowed_words]))

        return lowered

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



