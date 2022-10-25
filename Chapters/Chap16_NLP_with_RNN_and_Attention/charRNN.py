from typing import List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from pprint import pprint as pp


FILENAME = 'shakespeare.txt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BS = 256

def download_shakespeare():
    import requests

    shakespeare_url = "https://homl.info/shakespeare"
    r = requests.get(shakespeare_url)
    with open(FILENAME, 'w') as f:
        f.write(r.text)

def read_shakespeare():
    with open(FILENAME, 'r') as f:
        text = f.read()
    return text


def unique_char(text, lower=True):
    if lower:
        text = text.lower()
    return sorted(''.join(set(text)))


class Tokenize():
    def __init__(self, text, lower=True):
        self.word_index = unique_char(text, lower=lower)
        self.lower = lower
        self.t2s = {c: i for i, c in enumerate(self.word_index)}    # text 2 seq
        self.s2t = {i: c for i, c in enumerate(self.word_index)}    # seq 2 text
        self.document_count = len(text)

    def _lower(self, text: str):
        if self.lower:
            text = text.lower()
        return text

    def texts_to_sequences(self, text: List[str]):
        output = []
        for l in text:
            word = []
            for c in l:
                word.append(self.t2s[self._lower(c)])
            output.append(word)
        return output

    def sequences_to_texts(self, seq: List[List[int]]):
        output = []
        for s in seq:
            word = []
            for v in s:
                word.append(self.s2t[v])
            output.append(''.join(word))
        return output


class WordDataSet(Dataset):
    def __init__(self, text, window_length, word_index):
        super().__init__()
        self.text = text
        self.window_length = window_length
        self.word_index = word_index

    def __len__(self):
        return len(self.text) - self.window_length - 1

    def __getitem__(self, idx):
        one_hot = F.one_hot(torch.tensor(self.text[idx:idx+self.window_length+1]), num_classes = self.word_index)
        return one_hot[:-1].type(torch.float), one_hot[-1]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 128
        self.num_layers = 3
        self.gru = nn.GRU(39, self.hidden_size, dropout=0.2, batch_first=True, num_layers=self.num_layers)
        self.linear = nn.Linear(self.hidden_size, 39)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=DEVICE)
        output, hn = self.gru(x, h0)
        out = self.linear(output)
        return out


def train(model, train_dataloader, epochs, criterion, optimizer, valid_dataloader=None):
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_dataloader):
            if x.shape[0] != BS:
                continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            # forward
            logits = model(x)
            logits = logits[:, -1, :]
            loss = criterion(logits, y.argmax(dim=1))

            # Reset grads to 0
            optimizer.zero_grad()

            # backward
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                correct = torch.sum(logits.argmax(dim=1) == y.argmax(dim=1))
                print(f'epoch {epoch+1} / {epochs}, step {i+1}/{len(train_dataloader)}, loss = {loss.item():.4f}, acc = {correct/BS}')

        torch.save({'model_state_dict': model.state_dict()}, f'checkpoints/epoch_{epoch}_loss_{loss}.ckpt')
        if valid_dataloader is not None:
            with torch.no_grad():
                model = model.eval()
                for i, (x, y) in enumerate(valid_dataloader):
                    if x.shape[0] != BS:
                        continue
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    # forward
                    logits = model(x)
                    logits = logits[:, -1, :]
                    loss = criterion(logits, y.argmax(dim=1))
                    if (i+1) % 100 == 0:
                        correct = torch.sum(logits.argmax(dim=1) == y.argmax(dim=1))
                        print(f'epoch {epoch+1} / {epochs}, step {i+1}/{len(valid_dataloader)}, loss = {loss.item():.4f}, acc = {correct/BS}')
            model = model.train()


def preprocess(texts, tokenizer):
    X = tokenizer.texts_to_sequences(texts)
    one_hot = F.one_hot(torch.tensor(X, device=DEVICE), num_classes=len(tokenizer.word_index)).type(torch.float)
    return one_hot


def next_char(text, model, tokenizer, temp=1):
    X_new = preprocess([text], tokenizer)
    y_proba = model(X_new).softmax(dim=-1)[0, -1:, :]
    y_proba = y_proba.log() / temp
    s = y_proba.argmax(dim=-1)
    #m = Categorical(logits=y_proba)
    #s = m.sample()
    return tokenizer.sequences_to_texts([s.cpu().numpy().tolist()])[0]


def complete_text(text, model, tokenizer, n_chars=50):
    for _ in range(n_chars):
        text += next_char(text, model, tokenizer)
    return text


if __name__ == '__main__':
    #------------------------------------
    # Work with the data
    #------------------------------------
    #download_shakespeare()

    text = read_shakespeare()

    tokenizer = Tokenize(text, lower=True)
    pp(tokenizer.t2s)
    pp(tokenizer.texts_to_sequences(['First', 'word']))
    pp(tokenizer.sequences_to_texts([[18, 21, 30, 31, 32], [35, 27, 30, 16]]))

    [encoded] = np.array(tokenizer.texts_to_sequences([text]))

    #------------------------------------
    # Training Datasets
    #------------------------------------
    if 0:
        train_size = len(encoded) * 90 // 100
        valid_size = len(encoded) * 5 // 100
        n_steps = 100

        train_data = encoded[:train_size]
        train_ds = WordDataSet(train_data, n_steps, word_index = len(tokenizer.word_index))
        train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)

        valid_data = encoded[train_size : train_size + valid_size]
        valid_ds = WordDataSet(valid_data, n_steps, word_index = len(tokenizer.word_index))
        valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=False)

    #------------------------------------
    # Create the model and train
    #------------------------------------
    model = Model().to(DEVICE)

    if 0:
        learning_rate = 0.0001
        epochs = 5

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train(model, train_dl, epochs, criterion, optimizer, valid_dl)

    else:
        checkpoint = 'checkpoints/content/checkpoints/epoch_32_loss_1.4621447324752808.ckpt'

        ckpt = torch.load(checkpoint)['model_state_dict']
        model.load_state_dict(ckpt)

    #------------------------------------
    # Use the model to predict
    #------------------------------------

    # Simple example
    TEXT = ['Are you g']

    X_new = preprocess(TEXT, tokenizer)
    y_pred = model(X_new)[:,-1,:].argmax(dim=-1).cpu().numpy().tolist()
    pred_letter = tokenizer.sequences_to_texts([y_pred])
    print(f'Input: {TEXT}, next letter: {pred_letter}')

    # More text
    n = complete_text(TEXT[0], model, tokenizer, n_chars=200)
    print(n)
