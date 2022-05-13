from collections import defaultdict
import numpy as np
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from prep import *
from datetime import datetime
from bert import SentimentClassifier, create_data_loader


class SentimentMetaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentMetaClassifier, self).__init__()
        self.fc = nn.Linear(n_classes * 2, 64)
        self.fc2 = nn.Linear(self.bert.config.hidden_size, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return self.softmax(self.fc2(x))

    def reset_linear_weight(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)


class TwitterPredictionsDataset(Dataset):
    def __init__(self, predictions1, predictions2, labels):
        self.predictions1 = predictions1
        self.predictions2 = predictions2
        self.labels = labels

    def __len__(self):
        return len(self.predictions1)

    def __getitem__(self, item):
        label = self.labels[item]


        return {
          'prediction': torch.cat((predictions[1], predictions[2]), dim=1),
          'label': torch.tensor(label, dtype=torch.long)
        }


label_encodings = ["negative", "neutral", "positive"]

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SentimentClassifier(3, bertweet)
model.load_state_dict(torch.load("models/best_model_state_ensemble_(1).bin"))
model.to(device)

model1 = model.eval()


data_path = "data/Train.csv"

data = pd.read_csv(data_path)

data["sentiment"] = data["sentiment"].replace(label_encodings)

datasets = [[], []]

for class_index, group in data.groupby("sentiment"):
    x = data[data["sentiment"] == class_index]
    datasets[0].append(x[:int(len(x) / 2)])
    datasets[1].append(x[int(len(x) / 2):])

data = pd.concat(datasets[1], ignore_index=True)

with torch.no_grad():
    predictions = [[], [], []]
    for x in create_data_loader(data, tokenizer, 16):
        input_ids = x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)
        label = x["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions[0].append(outputs)
        predictions[2].append(label)


    model2 = SentimentClassifier(3, bertweet)
    model2.load_state_dict(torch.load("models/best_model_state_ensemble_(2).bin"))
    model2.to(device)
    model2 = model.eval()

    for x in create_data_loader(data, tokenizer, 16):
        input_ids = x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)
        outputs = model2(input_ids=input_ids, attention_mask=attention_mask)
        predictions[1].append(outputs)


data = pd.DataFrame(predictions)
loss_fn = nn.CrossEntropyLoss().to(device)

K_FOLDS = 10

model = SentimentMetaClassifier(len(label_encodings))

model.to(device)
model.reset_linear_weight()

EPOCH_COUNT = 10

optimizer = AdamW(model.parameters(), lr=1e-3)

def train_epoch(model, train_loader, loss_fn, optimizer, device, n):
    model = model.train()
    losses = []
    curr_correct = 0
    correct = 0
    total = 0
    batch = 0
    for x in train_loader:
        inputs = x["prediction"].to(device)
        labels = x["label"].to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        curr_correct += torch.sum(preds == labels)
        total += labels.size(dim=0)
        losses.append(loss.item())

        batch += 1

        if batch % 100 == 0:
            print(f"Batch {batch} Accuracy: {float(curr_correct)/total}")
            correct += curr_correct
            total = 0
            curr_correct = 0

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return float(correct + curr_correct) / n, np.mean(losses)


def evaluate(model, test_loader, loss_fn, device, n):
    model = model.eval()

    losses = []
    correct = 0
    with torch.no_grad():
        for x in test_loader:
            inputs = x["prediction"].to(device)
            labels = x["label"].to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct += torch.sum(preds == labels)
            losses.append(loss.item())

    return float(correct) / n, np.mean(losses)


BATCH_SIZE = 16

K_FOLDS = 10

def create_prediction_data_loader(predictions1, predictions2, labels, batch_size):
    dataset = TwitterPredictionsDataset(predictions1, predictions2, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)


def create_k_split_loaders(predictions1, predictions2, labels, k, batch_size):
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    for train_idx, test_idx in splitter.split(df["prediction"], df["label"]):
        yield create_prediction_data_loader(df.loc[train_idx], BATCH_SIZE), create_prediction_data_loader(df.loc[test_idx], BATCH_SIZE)


history = []

best_val_loss = math.inf

loaders = create_k_split_loaders(data, K_FOLDS, BATCH_SIZE)

for i, (train_loader, test_loader) in enumerate(loaders):
    print(f'Split {i + 1} of {K_FOLDS}')
    print('-' * 12)

    tracker = defaultdict(list)

    for epoch in range(EPOCH_COUNT):
        print(f'Epoch {epoch + 1} of {EPOCH_COUNT}')
        print('*' * 12)

        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, len(train_loader.dataset))

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = evaluate(model, test_loader, loss_fn, device, len(test_loader.dataset))

        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()

        tracker['train_acc'].append(train_acc)
        tracker['train_loss'].append(train_loss)
        tracker['val_acc'].append(val_acc)
        tracker['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f'models/best_model_state_meta_classifier.bin')
            best_val_loss = val_loss

    history.append(tracker)

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    model.bert.init_weights()
    model.reset_linear_weight()






