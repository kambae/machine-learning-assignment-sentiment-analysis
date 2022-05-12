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

MAX_LEN = 32


class TwitterDataset(Dataset):

    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]
        encoding = self.tokenizer(tweet.replace("\"", "").replace("&amp;", "&"), padding="max_length", truncation=True, max_length=self.max_len, return_tensors='pt', add_special_tokens=True)

        return {
          'tweet_text': tweet,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer,  batch_size):
    df = oversample_classes(df)
    dataset = TwitterDataset(df["text"].to_numpy(), df.sentiment.to_numpy(), tokenizer, MAX_LEN)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)


BATCH_SIZE = 16


def create_k_split_loaders(df, k, batch_size):
    splitter = StratifiedKFold(n_splits=k, shuffle=True)
    for train_idx, test_idx in splitter.split(df["text"], df["sentiment"]):
        yield create_data_loader(df.loc[train_idx], tokenizer, BATCH_SIZE), create_data_loader(df.loc[test_idx], tokenizer, BATCH_SIZE)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes, bert):
        super(SentimentClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)["pooler_output"]
        x = self.dropout(x)
        return self.fc(x)

    def reset_linear_weight(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)


def train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, n):
    model = model.train()
    losses = []
    curr_correct = 0
    correct = 0
    total = 0
    batch = 0
    for x in train_loader:
        input_ids = x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)
        labels = x["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
        scheduler.step()
        optimizer.zero_grad()

    return float(correct + curr_correct) / n, np.mean(losses)


def evaluate(model, test_loader, loss_fn, device, n):
    model = model.eval()

    losses = []
    correct = 0
    with torch.no_grad():
        for x in test_loader:
            input_ids = x["input_ids"].to(device)
            attention_mask = x["attention_mask"].to(device)
            labels = x["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct += torch.sum(preds == labels)
            losses.append(loss.item())

    return float(correct) / n, np.mean(losses)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

    label_encodings = {"negative": 0, "neutral": 1, "positive": 2}

    data_path = "data/Train.csv"

    data = pd.read_csv(data_path)

    data = undersample_classes(data, 5000)

    data["sentiment"] = data["sentiment"].replace(label_encodings)

    K_FOLDS = 10

    model = SentimentClassifier(len(label_encodings), bertweet)
    model.to(device)
    model.reset_linear_weight()

    loaders = create_k_split_loaders(data, K_FOLDS, BATCH_SIZE)

    EPOCH_COUNT = 10

    optimizer = AdamW(model.parameters(), lr=1e-5)

    loss_fn = nn.CrossEntropyLoss().to(device)
    history = []

    best_val_loss = 0

    for i, (train_loader, test_loader) in enumerate(loaders):
        print(f'Split {i + 1} of {K_FOLDS}')
        print('-' * 12)
        total_steps = len(train_loader) * EPOCH_COUNT
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        tracker = defaultdict(list)

        for epoch in range(EPOCH_COUNT):
            print(f'Epoch {epoch + 1} of {EPOCH_COUNT}')
            print('*' * 12)

            train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device, len(train_loader.dataset))

            print(f'Train loss {train_loss} accuracy {train_acc}')

            val_acc, val_loss = evaluate(model, test_loader, loss_fn, device, len(test_loader.dataset))

            print(f'Val   loss {val_loss} accuracy {val_acc}')
            print()

            tracker['train_acc'].append(train_acc)
            tracker['train_loss'].append(train_loss)
            tracker['val_acc'].append(val_acc)
            tracker['val_loss'].append(val_loss)

            if val_loss > best_val_loss:
                torch.save(model.state_dict(), f'models/best_model_state_4.bin')
                best_val_loss = val_acc

        history.append(tracker)

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        model.bert.init_weights()
        model.reset_linear_weight()

    avg_acc = 0
    avg_loss = 0
    for tracker in history:
        avg_acc += tracker["val_acc"][-1] / K_FOLDS
        avg_loss += tracker["val_loss"][-1] / K_FOLDS

    print(f'AVERAGE FINAL ACCURACY {avg_acc}\nAVERAGE FINAL LOSS {avg_loss}')

