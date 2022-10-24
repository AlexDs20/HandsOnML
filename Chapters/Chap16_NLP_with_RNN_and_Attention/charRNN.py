from typing import List
import numpy as np
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
        self.unique = unique_char(text, lower=lower)
        self.lower = lower
        self.t2s = {c: i for i, c in enumerate(self.unique)}    # text 2 seq
        self.s2t = {i: c for i, c in enumerate(self.unique)}    # seq 2 text

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



if __name__ == '__main__':
    #download_shakespeare()

    text = read_shakespeare()

    tokenizer = Tokenize(text, lower=True)
    pp(tokenizer.t2s)
    pp(tokenizer.texts_to_sequences(['First', 'word']))
    pp(tokenizer.sequences_to_texts([[18, 21, 30, 31, 32], [35, 27, 30, 16]]))

    [encoded] = np.array(tokenizer.texts_to_sequences([text]))
    print(encoded.shape)
