import pathlib
import pickle
from datetime import datetime

import pandas as pd
from utils import count_ngrams, create_model

PROCESSED_DATA_DIR = pathlib.Path('../data/processed/')


def read_files():
    so = pd.read_csv(PROCESSED_DATA_DIR / 'tokenized.csv')
    so = so.loc[so.text.dropna().index]
    return so


def train_test_split(so, sample=True, random_state=0):
    train = so.query("category != 'title'")
    test = so.query("category == 'title'")
    if sample:
        train = train.sample(1000, random_state=random_state)
        test = test.sample(50, random_state=random_state)
    return train, test


def fit(train, n=3, save_model=False):
    vocab_set = set(' '.join(train.text.tolist()))
    freqs = count_ngrams(train.text.tolist(), n)
    model = create_model(freqs, len(vocab_set))
    if save_model:
        print("Saving model as pickle file")
        timestamp = datetime.now()
        pickle.dump(model, open(f"model_n{n}_{timestamp}.p", "wb"))
    return model


def main():
    so = read_files()
    train, test = train_test_split(so)

    # fit(train)


if __name__ == "__main__":
    main()
