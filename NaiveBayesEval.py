from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
from prep import *
from sklearn.model_selection import KFold, StratifiedKFold

if __name__ == "__main__":
    data_path = "data/Train.csv"
    test_data_path = "data/Test.csv"

    data = pd.read_csv(data_path)
    test_data = pd.read_csv(test_data_path)

    data = undersample_classes(data)

    # note: tfidf probably perfoming better in our test due to comment below - the actual pred has a lot of non existent words which will skew tfidf values
    # note: tfidf is returning far more neutrals

    # accs = []

    ngrams = [1, 2, 3]
    ks = [100, 500, 1000, 2000]

    kf = StratifiedKFold(n_splits=10, shuffle=True)

    for ngram in ngrams:
        for k in ks:

            accs = []
            for train_i, test_i in kf.split(data, data["sentiment"]):
                prep = BagOfWords(k=k, vectoriser="count", n_gram=ngram)
                train = data.iloc[train_i]
                test = data.iloc[test_i]
                train_x, train_y = prep.train_prep(train)
                test_x, test_y = prep.pred_prep(test)

                model = MultinomialNB()
                model.fit(train_x, train_y)

                test_pred_y = model.predict(test_x)
                accs.append(accuracy_score(test_y, test_pred_y))

            # print("Accuracy:", accs)
            print("Ngram", ngram)
            print("K", k)
            print("Average Accuracy", np.mean(accs))
            print("\n")

    prep = BagOfWords(k=500, vectoriser="count")
    train_x, train_y = prep.train_prep(data)

    model = MultinomialNB()
    model.fit(train_x, train_y)


    pred_x = prep.pred_prep(test_data)[0]
    out_pred_y = model.predict(pred_x)
    # print(pred_y)
    output_pred_csv(test_data, out_pred_y)