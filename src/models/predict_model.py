from train_model import fit, train_test_split, read_files

from utils import sentence_perplexity
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import ngrams
from nltk.tokenize import WordPunctTokenizer
from utils import PADDING


DATASET_SIZE = 2000


def is_unknown(model, words):
    prefix, target = words[:-1], words[-1]
    return model[prefix][target] == 0


def compute_missing_prob(model, words, vocab_set):
    prefix = words[:-1]
    token_probs = model[prefix]
    missing_prob_total = 1.0 - sum(token_probs.values())
    missing_prob_total = max(0, missing_prob_total)  # prevent rounding errors
    missing_prob_total = missing_prob_total / max(1, len(vocab_set)
                                                  - len(token_probs))
    return missing_prob_total


def sentence_prob(model, sentence, vocab_set, n=3):
    """Only handles non OOV words"""
    sentence_split = sentence.split()
    num_words = len(sentence_split)
    trigrams = ngrams(
        sentence_split, n,
        pad_left=True,
        pad_right=True,
        left_pad_symbol=PADDING.left_pad_symbol,
        right_pad_symbol=PADDING.right_pad_symbol
    )
    probs = []
    for words in trigrams:
        prefix, target = words[:n-1], words[n-1]
        if not is_unknown(model, words):
            probs.append(np.log(model[prefix][target]))
    return np.exp(np.sum((probs))*(-1/num_words))


def perplexity(model, sentence, vocab_set, min_logprob=np.log(10 ** -50.)):
    """
    :param min_logprob: if log(P(w | ...)) is smaller than min_logprop,
        set it equal to min_logrob
    :returns: perplexity of a sentence - scalar
    """
    return sentence_prob(model, sentence, vocab_set)


def perplexity_laplace(sentence, counts, ngrams_degree=3, delta=1):
    """
    Handle OOV words and generalize perplexity to ngrams
    Credit: Manning liveProject example solution
    """
    sentence = WordPunctTokenizer().tokenize(sentence.lower())
    N = len(sentence)
    logprob = 0
    for ngram in ngrams(
          sentence,
          n=ngrams_degree,
          pad_right=True,
          pad_left=True,
          left_pad_symbol="<s>",
          right_pad_symbol="</s>"):
        prefix = ngram[:ngrams_degree-1]
        token = ngram[ngrams_degree-1]
        if prefix in list(counts.keys()):
            total = sum(counts[prefix].values())
            if token in counts[prefix].keys():
                # normal calculation
                logprob += np.log(
                    (counts[prefix][token] + delta) / (total + delta * N))
            else:
                logprob += np.log((delta) / (total + delta * N))
        else:
            logprob += -np.log(N)
    return np.exp(-logprob / N)


def corpus_perplexity(perplexities, vocab_size):
    return np.exp(np.sum(np.log(perplexities))*(-1/vocab_size))


def predict(model, test, ngrams_degree, vocab_set):
    tqdm.pandas()
    N = len(vocab_set)
    # perplexities = test.text.progress_apply(
        # lambda x: perplexity_laplace(x, counts, ngrams_degree))
    perplexities = test.text.progress_apply(
        lambda x: perplexity(model, x, vocab_set, ngrams_degree))
    return corpus_perplexity(perplexities, N)


def main():
    so = read_files()
    train, test = train_test_split(so, DATASET_SIZE)
    vocab_set = set(' '.join(train.text.tolist()))

    ngram_perplexity = dict()
    for ngrams_degree in range(1, 6):
        model, counts = fit(train, n=ngrams_degree)
        ngram_perplexity[ngrams_degree] = predict(
            model, test, ngrams_degree, vocab_set)
    print(ngram_perplexity)


if __name__ == "__main__":
    main()
