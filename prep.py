import numpy as np
import pandas as pd
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import csv
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
import re

output_path = "pred.csv"

# report can talk about: ngrams, data cleaning, select k best
class BagOfWords():

    def __init__(self, k=None, vectoriser="tfidf"):
        if vectoriser == "count":
            self.vectoriser = CountVectorizer(max_features=2000)
        else:
            self.vectoriser = TfidfVectorizer(max_features=2000)
        self.k = k

    def train_prep(self, data):
        text = data["text"]
        y = data["sentiment"].to_numpy()
        # clean text
        text = self.clean(text)
        X = self.vectoriser.fit_transform(text)

        if self.k is not None:
            k_best = SelectKBest(f_classif, k=self.k)
            k_best.fit(X.toarray(), y)
            indicies = k_best.get_support(indices=True)
            vocabulary = [self.vectoriser.get_feature_names_out()[i] for i in indicies]
            self.vectoriser.vocabulary = vocabulary
            X = self.vectoriser.transform(text)

            # print(vocabulary)

        return X.toarray(), y

    def pred_prep(self, data):
        text = data["text"]
        text = self.clean(text)
        y = data["sentiment"] if "sentiment" in data.columns else None

        return self.vectoriser.transform(text).toarray(), y

    def clean(self, data_text):
        lowered = data_text.str.strip().str.lower().str.strip('"')

        # lowered.apply(lambda x: print(re.match(r'https?://t.co/[a-zA-Z0-9]*\b', x)))

        lowered = lowered.apply(lambda x: re.sub(r'@(\w{1,15})\b', "USERNAME", x))
        lowered = lowered.apply(lambda x: re.sub(r'https?://t.co/[a-zA-Z0-9]*\b', "URL", x))

        # lowered = lowered.apply(lambda x: x.replace("#", ""))
        # lowered = lowered.apply(lambda x: x.replace('"', ""))

        # # todo: make optional through parameter
        # stemmer = PorterStemmer()
        # tk = TweetTokenizer()
        # allowed_words = set(words.words("en")) - set(stopwords.words("english"))
        # lowered.apply(lambda x: " ".join([stemmer.stem(i) for i in tk.tokenize(x) if i in allowed_words]))

        return lowered

def undersample_classes(data):
    nmin = data["sentiment"].value_counts().min()
    return data.groupby("sentiment").apply(lambda x: x.sample(nmin)).reset_index(drop=True)

def oversample_classes(data):
    nmax = data["sentiment"].value_counts().max()
    lst = [data]
    for class_index, group in data.groupby("sentiment"):
        lst.append(group.sample(nmax - len(group), replace=True))
    return pd.concat(lst)

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

    bow = BagOfWords()
    bow.train_prep(data)
    text_uncleaned = data["text"]
    text_cleaned = bow.clean(data["text"])

    for i in range(0, len(text_uncleaned)):
        print(text_uncleaned[i])
        print(text_cleaned[i])



