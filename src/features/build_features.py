import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from nltk.utils import ngrams


PROCESSED_DATA_DIR = Path('../../data/processed/')
tokenized = pd.read_csv(PROCESSED_DATA_DIR / 'tokenized.csv')


def train_test_split(tokenized):
    """Returns a train, test tuple of dataframes"""
    train = tokenized.query("category != 'title'")
    test = tokenized.query("category == 'title'")
    return train, test



