from train_model import fit, train_test_split, read_files

from utils import sentence_perplexity
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import ngrams
from utils import PADDING



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


def sentence_prob(model, sentence, vocab_set, min_logprob=np.log(10 ** -50.)):
    sentence_split = sentence.split()
    num_words = len(sentence_split)
    trigrams = ngrams(
        sentence_split, 3,
        pad_left=True,
        pad_right=True,
        left_pad_symbol=PADDING.left_pad_symbol,
        right_pad_symbol=PADDING.right_pad_symbol
    )
    probs = []
    for words in trigrams:
        prefix, target = words[:-1], words[-1]
        if is_unknown(model, words):
            missing_prob_total = compute_missing_prob(model, words, vocab_set)
            if (missing_prob_total == 0 or np.log(missing_prob_total)
                    < min_logprob):
                probs.append(min_logprob)
            else:
                probs.append(np.log(missing_prob_total))
        else:
            probs.append(np.log(model[prefix][target]))
    # return product(probs)
    return np.exp(np.sum((probs))*(-1/num_words))


def perplexity(model, sentence, vocab_set, min_logprob=np.log(10 ** -50.)):
    """
    :param min_logprob: if log(P(w | ...)) is smaller than min_logprop,
        set it equal to min_logrob
    :returns: perplexity of a sentence - scalar
    """
    return sentence_prob(model, sentence, vocab_set)


def corpus_perplexity(perplexities, vocab_size):
    return np.prod(perplexities)**(-1/vocab_size)


def predict(model, test, vocab_set):
    tqdm.pandas()
    N = len(vocab_set)
    perplexities = test.text.progress_apply(
        lambda x: perplexity(model, x, vocab_set))
    return corpus_perplexity(perplexities, N)


def main():
    so = read_files()
    train, test = train_test_split(so)
    vocab_set = set(' '.join(train.text.tolist()))

    ngram_perplexity = dict()
    for n in range(1, 6):
        model = fit(train, n=n)
        ngram_perplexity[n] = predict(model, test, vocab_set)
    print(ngram_perplexity)


if __name__ == "__main__":
    main()
