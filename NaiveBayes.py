from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
from prep import *

if __name__ == "__main__":
    data_path = "data/Train.csv"
    test_data_path = "data/Test.csv"

    data = pd.read_csv(data_path)
    test_data = pd.read_csv(test_data_path)

    data = undersample_classes(data)

    prep = BagOfWords(k=500, vectoriser="count")
    # note: tfidf probably perfoming better in our test due to comment below - the actual pred has a lot of non existent words which will skew tfidf values
    # note: tfidf is returning far more neutrals

    train, test = train_test_split(data, test_size=0.1)
    train_x, train_y = prep.train_prep(train)
    test_x, test_y = prep.pred_prep(test)

    pred_x = prep.pred_prep(test_data)[0]

    model = MultinomialNB()
    model.fit(train_x, train_y)

    test_pred_y = model.predict(test_x)
    print("Accuracy:", accuracy_score(test_y, test_pred_y))

    out_pred_y = model.predict(pred_x)
    # print(pred_y)
    output_pred_csv(test_data, out_pred_y)