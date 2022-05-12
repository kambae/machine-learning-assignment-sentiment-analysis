from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
from prep import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import BaggingClassifier


if __name__ == "__main__":
    data_path = "data/Train.csv"
    test_data_path = "data/Test.csv"

    data = pd.read_csv(data_path)
    test_data = pd.read_csv(test_data_path)

    data = undersample_classes(data)

    accs = []

    kf = KFold(n_splits=10, shuffle=True)
    for train_i, test_i in kf.split(data, data["sentiment"]):
        prep = BagOfWords(k=1000, vectoriser="count")
        train = data.iloc[train_i]
        test = data.iloc[test_i]
        train_x, train_y = prep.train_prep(train)
        test_x, test_y = prep.pred_prep(test)

        model = BaggingClassifier(base_estimator=svm.LinearSVC(C=0.01), n_estimators=10, max_samples=0.5)
        # model = svm.LinearSVC(C=0.01)
        model.fit(train_x, train_y)

        test_pred_y = model.predict(test_x)
        accs.append(accuracy_score(test_y, test_pred_y))

    print("Accuracy:", accs)
    print("Average Accuracy", np.mean(accs))

    prep = BagOfWords(k=1000, vectoriser="count")
    train_x, train_y = prep.train_prep(data)

    model = svm.LinearSVC()
    model.fit(train_x, train_y)

    pred_x = prep.pred_prep(test_data)[0]
    out_pred_y = model.predict(pred_x)
    # print(pred_y)
    output_pred_csv(test_data, out_pred_y)