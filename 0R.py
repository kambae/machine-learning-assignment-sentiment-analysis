from sklearn.dummy import DummyClassifier
from prep import bag_of_words
import pandas as pd

if __name__ == "__main__":
    data_path = "data/Train.csv"
    test_data_path = "data/Test.csv"

    data = pd.read_csv(data_path)
    test_data = pd.read_csv(test_data_path)

    train_x, train_y = bag_of_words(data)
    test_x, _ = bag_of_words(test_data)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(train_x, train_y)

    print(model.predict(test_x))