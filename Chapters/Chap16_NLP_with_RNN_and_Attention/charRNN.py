from typing import List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from pprint import pprint as pp


FILENAME = 'shakespeare.txt'

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
        self.gru = nn.GRU(39, 128, dropout=0.2, batch_first=True, num_layers=2)
        self.h0 = torch.zeros(2, 32, 128)

        self.linear = nn.Linear(128, 39)

    def forward(self, x):
        output, hn = self.gru(x, self.h0)
        out = self.linear(output)
        return out


def train(model, train_dataloader, epochs, criterion, optimizer, valid_dataloader=None):
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_dataloader):
            # forward
            logits = model(x)
            loss = criterion(logits, y)

            # Reset grads to 0
            optimizer.zero_grad()

            # backward
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'epoch {epoch+1} / {epochs}, step {i+1}/{len(train_dataloader)}, loss = {loss.item():.4f}')

        if valid_data_loader is not None:
            with torch.no_grad():
                for i, (x, y) in enumerate(valid_dataloader):
                    # forward
                    logits = model(x)
                    loss = criterion(logits, y)
                    if (i+1) % 100 == 0:
                        print(f'epoch {epoch+1} / {epochs}, step {i+1}/{len(valid_dataloader)}, loss = {loss.item():.4f}')


if __name__ == '__main__':
    #download_shakespeare()

    text = read_shakespeare()

    tokenizer = Tokenize(text, lower=True)
    pp(tokenizer.t2s)
    pp(tokenizer.texts_to_sequences(['First', 'word']))
    pp(tokenizer.sequences_to_texts([[18, 21, 30, 31, 32], [35, 27, 30, 16]]))

    [encoded] = np.array(tokenizer.texts_to_sequences([text]))

    train_size = len(encoded) * 90 // 100
    n_steps = 100

    train_data = encoded[:train_size*len(encoded)]
    train_ds = WordDataSet(train_data, n_steps, word_index = len(tokenizer.word_index))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = Model()
    x, y = next(iter(train_dl))
    out = model(x)
    print(out.shape)

    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_dl, 5, criterion, optimizer)
