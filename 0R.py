from sklearn.dummy import DummyClassifier
from prep import BagOfWords, output_pred_csv, undersample_classes
import pandas as pd
from sklearn.metrics import *

if __name__ == "__main__":
    data_path = "data/Train.csv"
    test_data_path = "data/Test.csv"

    data = pd.read_csv(data_path)
    data = undersample_classes(data)
    test_data = pd.read_csv(test_data_path)

    prep = BagOfWords()
    train_x, train_y = prep.train_prep(data)
    test_x = prep.pred_prep(test_data)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_x, train_y)
    train_pred_y = model.predict(train_x)

    print("Accuracy", accuracy_score(train_y, train_pred_y))
    print("F1 score", f1_score(train_y, train_pred_y, average="macro"))

    pred_y = model.predict(test_x)
    # print(pred_y)
    output_pred_csv(test_data, pred_y)