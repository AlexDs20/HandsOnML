from typing import List
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, text, window_length):
        super().__init__()
        self.text = text
        self.window_length = window_length

    def __len__(self):
        return len(self.text) - self.window_length - 1

    def __getitem__(self, idx):
        return self.text[idx:idx+self.window_length], self.text[idx+self.window_length]



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

    train_ds = WordDataSet(encoded[:train_size*len(encoded)], n_steps)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
    input, t = next(iter(train_dl))
    for idx, (input, t) in enumerate(train_dl):
        print(tokenizer.sequences_to_texts([input[0,:].numpy()]))
        print(tokenizer.sequences_to_texts([[t[0].numpy().tolist()]]))
        if idx==5:
            break
