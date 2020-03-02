# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import re
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from nltk.tokenize import word_tokenize


code_exp = re.compile(r"<pre[^>]*>.+?</pre>", re.DOTALL)
url_exp = re.compile(r"(?P<url>(http\S+))")
start_tag = re.compile(r"<[a-z][^>]*>")
end_tag = re.compile(r"</[a-z]+>")
latex_exp = re.compile(r"(?P<latex>(\$\S+\$))")
latex_exp2 = re.compile(r"\${2}.+\${2}")
newline_exp = re.compile(r"(?P<newline>(\n+))")
digit_exp = re.compile(r"(?P<digit>\d+\.*\d+)")


def regex(so):
    patterns = [
        code_exp,
        url_exp,
        start_tag,
        end_tag,
        latex_exp,
        latex_exp2,
        # newline_exp,
        digit_exp
    ]

    for pattern in patterns:
        so.loc[:, 'text'] = so.text.replace(pattern, '', regex=True)
    so.loc[:, 'text'] = so.text.replace(newline_exp, ' ', regex=True)
    return so


def remove_missing(so):
    so = so.replace('', np.NaN)
    so = so.dropna(subset=['text'])
    return so


def remove_outliers(so):
    """Remove the top and bottom 10 percentile of text by char length"""
    so['text_lengths'] = so.text.str.len()
    so = so[so.text_lengths < so.text_lengths.quantile(0.90)]
    so = so[so.text_lengths > so.text_lengths.quantile(0.10)]
    return so.drop(columns='text_lengths')


def pandas_tokenize(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    print(input_filepath)

    so = pd.read_csv(Path(input_filepath) / 'stackexchange_812k.csv')
    so = regex(so)
    so = remove_missing(so)
    so = remove_outliers(so)
    so.loc[:, 'text'] = so.text.apply(pandas_tokenize)
    so.to_csv(Path(output_filepath) / 'tokenized.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
