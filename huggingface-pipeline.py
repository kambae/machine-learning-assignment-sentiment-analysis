import numpy as np
from transformers import pipeline
import pandas as pd
from prep import *

SentimentClassifier = pipeline("sentiment-analysis", model="vinai/bertweet-base")

data_path = "data/Train.csv"
data = pd.read_csv(data_path)

result = np.array([sentiment["label"].lower() for sentiment in SentimentClassifier(BagOfWords.clean(data["text"]).tolist()[:16])])

# accuracy = np.sum((np.array([sentiment["label"].lower() for sentiment in SentimentClassifier(BagOfWords.clean(data["text"]).tolist()[:128])]) == np.array(data["sentiment"])[:128]))/len(data)

print(list(zip(result, data["text"][:16], data["sentiment"][:16])))