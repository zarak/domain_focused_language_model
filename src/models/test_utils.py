import pickle

import numpy as np
from src.models.utils import (product,
                              sentence_perplexity,
                              count_ngrams,
                              create_model)

model = pickle.load(open("model_2020-03-09 11:01:38.125971.p", "rb"))
sentences = [
    'test this planet hurrah test this planet',
    'another test this planet',
    'test this spoon'
]
vocab_set = set(' '.join(sentences))
vocab_size = len(vocab_set)


def test_product_zero():
    a = np.array(range(10))
    z = product(a)
    assert np.isclose(z, np.prod(a))


def test_product():
    a = np.array(range(1, 20))
    z = product(a)
    assert np.isclose(z, np.prod(a))


def test_product_underflow():
    a = np.array(range(1, 20)) / 10**6
    z = product(a)
    assert np.isclose(z, np.prod(a))


def test_count_ngrams():
    freqs = count_ngrams(sentences, n=3)
    assert freqs[('test', 'this')]['planet'] == 3
    assert freqs[('<s>', 'test')]['this'] == 2


def test_create_model():
    freqs = count_ngrams(sentences, n=3)
    model = create_model(freqs, vocab_size)
    num_prefix_token = 3
    num_prefix_only = 4
    mle_prob = num_prefix_token / num_prefix_only
    assert np.isclose(model[('test', 'this')]['planet'], mle_prob)


def test_create_model_unknown_word():
    freqs = count_ngrams(sentences, n=3)
    model = create_model(freqs, vocab_size)
    num_prefix_token = 0
    num_prefix_only = 4
    mle_prob = num_prefix_token / num_prefix_only
    assert np.isclose(model[('test', 'this')]['widget'], mle_prob)


def test_create_model_certain():
    freqs = count_ngrams(sentences, n=3)
    model = create_model(freqs, vocab_size)
    num_prefix_token = 1
    num_prefix_only = 1
    mle_prob = num_prefix_token / num_prefix_only
    assert np.isclose(model[('planet', 'hurrah')]['test'], mle_prob)


def test_create_model_left_padding():
    freqs = count_ngrams(sentences, n=3)
    model = create_model(freqs, vocab_size)
    num_prefix_token = 2
    num_prefix_only = 3
    mle_prob = num_prefix_token / num_prefix_only
    assert np.isclose(model[('<s>', '<s>')]['test'], mle_prob)


def test_create_model_right_padding():
    freqs = count_ngrams(sentences, n=3)
    model = create_model(freqs, vocab_size)
    num_prefix_token = 2
    num_prefix_only = 3
    mle_prob = num_prefix_token / num_prefix_only
    assert np.isclose(model[('this', 'planet')]['</s>'], mle_prob)


def test_sentence_perplexity():
    freqs = count_ngrams(sentences, n=3)
    model = create_model(freqs, vocab_size)

    sentence = "test this planet hurrah test this spoon"
    start_test = model[('<s>', '<s>')]['test']
    start_testthis = model[('<s>', 'test')]['this']

    testthis_planet = model[('test', 'this')]['planet']
    thisplanet_hurrah = model[('this', 'planet')]['hurrah']
    planethurrah_test = model[('planet', 'hurrah')]['test']
    hurrahtest_this = model[('hurrah', 'test')]['this']
    testthis_spoon = model[('test', 'this')]['spoon']

    end_thisspoon = model[('this', 'spoon')]['</s>']
    end_spoon = model[('spoon', '</s>')]['</s>']

    start_trigrams = [start_test, start_testthis]
    end_trigrams = [end_thisspoon, end_spoon]
    middle_trigrams = [
            testthis_planet,
            thisplanet_hurrah,
            planethurrah_test,
            hurrahtest_this,
            testthis_spoon,
    ]
    all_probs = start_trigrams + middle_trigrams + end_trigrams

    prod_prob = np.prod(all_probs)

    perp = prod_prob ** (-1/len(sentence.split()))
    print('perp', perp)
    assert (sentence_perplexity(model, sentence)) == perp
