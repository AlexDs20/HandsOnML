#! python3
import os
import tarfile
import urllib.request
from zlib import crc32
from sklearn import model_selection
from scipy import stats

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def fetch_data(url, data_path, data):
    os.makedirs(data_path, exist_ok=True)
    tgz_path = os.path.join(data_path, data)
    urllib.request.urlretrieve(url, tgz_path)
    data_tgz = tarfile.open(tgz_path)
    data_tgz.extractall(path=data_path)
    data_tgz.close()


def load_data(data_path):
    return pd.read_csv(data_path)


def display_info(data):
    print('Head: \n', data.head(), '\n')
    print('Info: \n', data.info(), '\n')
    print('Describe: \n', data.describe(), '\n')


def plot_hist(data):
    data.hist(bins=50, figsize=(10, 8))
    plt.show()
    data["ocean_proximity"].hist()
    plt.show()


def split_train_test(data, test_ratio):
    """
    For always having the same data:
        np.random.seed(42)
    However, this does not work if updating the dataset
    -> use split by id
    """
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    """
    If new data used, append them at the end of data!
    """
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def stratified_split(data, cat, bins, test_size):
    lab = list(range(len(bins)-1))
    temp_cat = "temp"
    data[temp_cat] = pd.cut(data[cat],
                            bins=bins,
                            labels=lab)

    split = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_index, test_index in split.split(data, data[temp_cat]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop(temp_cat, axis=1, inplace=True)

    return strat_train_set, strat_test_set


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


def confidence_interval(confidence, predictions, labels):
    squared_errors = (predictions - labels) ** 2
    interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                        loc=squared_errors.mean(),
                                        scale=stats.sem(squared_errors)))
    return interval
