from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
from prep import BagOfWords, output_pred_csv

if __name__ == "__main__":
    data_path = "data/Train.csv"
    test_data_path = "data/Test.csv"

    data = pd.read_csv(data_path)
    test_data = pd.read_csv(test_data_path)

    prep = BagOfWords(k=100, vectoriser="tfidf")

    # todo: currently featursises words that occur in test but not train. This should not occur!
    train_x, test_x, train_y, test_y = train_test_split(*prep.train_prep(data), test_size=0.1)
    pred_x = prep.pred_prep(test_data)

    model = MultinomialNB()
    model.fit(train_x, train_y)

    test_pred_y = model.predict(test_x)
    print("Accuracy:", accuracy_score(test_y, test_pred_y))

    out_pred_y = model.predict(pred_x)
    # print(pred_y)
    output_pred_csv(test_data, out_pred_y)