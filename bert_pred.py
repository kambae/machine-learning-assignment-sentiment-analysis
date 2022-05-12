import torch
from bert import SentimentClassifier
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from prep import *

label_encodings = ["negative", "neutral", "positive"]

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SentimentClassifier(3, bertweet)
model.load_state_dict(torch.load("models/best_model_state_4.bin"))
model.to(device)

model = model.eval()

data_path = "data/Test.csv"

data = pd.read_csv(data_path)

with torch.no_grad():
    predictions = []
    for sample in data["text"]:
        encoding = tokenizer(sample.replace("\"", "").replace("&amp;", "&"), padding="max_length",
                                 truncation=True, max_length=64, return_tensors='pt', add_special_tokens=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = torch.max(outputs, dim=1)
        predictions.append(label_encodings[pred.item()])
    output_pred_csv(data, predictions)


