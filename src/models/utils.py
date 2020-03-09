from collections import namedtuple, defaultdict, Counter
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from nltk import ngrams


Padding = namedtuple('Padding', ['left_pad_symbol', 'right_pad_symbol'])
PADDING = Padding("<s>", "</s>")


def count_ngrams(lines, n):
    """
    Count how many times each word occured after (n - 1) previous words
    :param lines: an iterable of strings with space-separated tokens
    """
    counts = defaultdict(Counter)
    for line in tqdm(lines, desc='Sentences'):
        words = ngrams(line.split(), n,
                       pad_left=True,
                       pad_right=True,
                       left_pad_symbol=PADDING.left_pad_symbol,
                       right_pad_symbol=PADDING.right_pad_symbol)
        for word in words:
            prefix, target = word[:-1], word[-1]
            counts[prefix][target] += 1
    return counts


def create_model(freqs, vocab_size, delta=1):
    """
    Transform the counts to probabilities.
    Default has no Laplace smoothing.
    """
    freqs = deepcopy(freqs)
    for prefix in tqdm(freqs):
        token_counts = freqs[prefix]
        total_count = float(sum(token_counts.values()) + delta * vocab_size)
        for w3 in freqs[prefix]:
            token_counts[w3] = (token_counts[w3] + delta) / total_count
    return freqs


def sentence_perplexity(model, sentence):
    words = sentence.split()
    num_words = len(words)
    trigrams = ngrams(
        words, 3,
        pad_left=True,
        pad_right=True,
        left_pad_symbol=PADDING.left_pad_symbol,
        right_pad_symbol=PADDING.right_pad_symbol
    )
    probs = []
    for words in trigrams:
        prefix, target = words[:-1], words[-1]
        probs.append(model[prefix][target])
    # return product(probs)
    return np.prod(probs)**(-1/num_words)


def generate_text(seed, model, max_length=10):
    """Takes a bigram as input and generates the next token"""
    assert len(seed) < max_length, \
        "Max length must be greater than the length of the seed"
    sentence_finished = False

    while (not sentence_finished) and len(seed) <= max_length:
        probs = list(model[tuple(seed[-2:])].values())
        words = list(model[tuple(seed[-2:])].keys())
        seed.append(np.random.choice(words, p=probs))
        if seed[-2:] == ['</s>', '</s>']:
            sentence_finished = True
    return ' '.join([t for t in seed if t not in PADDING])


def product(array):
    """Numerically stable product"""
    if any([num == 0 for num in array]):
        return 0
    return np.exp(np.sum(np.log(array)))
