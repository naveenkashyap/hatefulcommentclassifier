import re
import string
import collections
import itertools

from nltk.stem.porter import PorterStemmer


def iter_ngram(n, words):
    """
    Iterate over n-grams.

    :param n: the "n"-gram
    :param words: an iterable of words
    :yield: the ngrams

    >>> list(iter_ngram(1, ['hello', 'world']))
    [('hello',), ('world',)]
    >>> list(iter_ngram(2, ['hello', 'world']))
    [('hello', 'world')]
    """
    words = iter(words)
    cache = collections.deque(maxlen=n)
    try:
        for _ in range(n - 1):
            cache.append(next(words))
    except StopIteration:
        return
    try:
        for w in words:
            cache.append(next(words))
            yield tuple(cache)
    except StopIteration:
        pass

def count_ngram(n, ngrams_list):
    """
    Count occurrences of
    :param n: the "n"-gram
    :param ngrams_list: list of ngrams
    :return: a dict of ngram-to-count

    >>> cnt = count_ngram(1, [[('hello',), ('world',)], [('again',)]])
    """
    c = collections.Counter()
    for words in corpus:
        for ng in iter_ngram(n, words):
            c[ng] += 1
    c = dict(c)
    return c

def preprocess_comments(corpus):
    """
    Remove capital letters, remove punctuations, split into words, stem the words.

    :param corpus: an iterable of comments
    :return: a iterable of processed comments
    """
    puncset = set(string.punctuation)
    stemmer = PorterStemmer()

    comments = iter(''.join(c for c in x if c not in puncset) for x in comments)
    comments = iter(x.lower() for x in comments)
    comments = iter(re.sub(r'([a-z])([0-9])', r'\1 \2', x) for x in comments)
    comments = iter(re.sub(r'([0-9])([a-z])', r'\1 \2', x) for x in comments)
    comments = map(str.split, comments)
    comments = iter(list(map(stemmer.stem, x)) for x in comments)
    return comments
