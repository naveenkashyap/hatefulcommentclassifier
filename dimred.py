from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn
import pickle
import itertools
import multiprocessing
from functools import partial
import collections
import string
import sys
import re
import random
import numpy as np
import pandas as pd
import pdb
from surprise import SVD as SGDSVD
import scipy.sparse
import logging
logging.basicConfig(level=logging.INFO, filename='launch.log')


LONG_WORD_TOKEN = "<LONG_WORD>"


def get_train_data():
    return pd.read_csv("train.csv")


def get_test_data():
    test_comments = pd.read_csv("test.csv")
    test_labels = pd.read_csv("test_labels.csv")
    return test_comments, test_labels


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


def stem_comment(stemmer, comment):
    stemmed = []
    for word in comment:
        try:
            stemmed.append(stemmer.stem(word))
        except RecursionError:
            logging.warning("stemmer recursion issue")
            stemmed.append(LONG_WORD_TOKEN)
    return ' '.join(stemmed)


def preprocess_comments(corpus):
    """
    Remove capital letters, remove punctuations, split into words, stem the words.

    :param corpus: an iterable of comments
    :return: a iterable of processed comments
    """
    stemmer = PorterStemmer()
    puncset = set(string.punctuation)
    comments = corpus

    comments = iter(''.join(c for c in x if c not in puncset)
                    for x in comments)
    comments = iter(x.lower() for x in comments)
    comments = iter(re.sub(r'([a-z])([0-9])', r'\1 \2', x) for x in comments)
    comments = iter(re.sub(r'([0-9])([a-z])', r'\1 \2', x) for x in comments)
    comments = map(str.split, comments)
    with multiprocessing.Pool() as pool:
        ret = list(pool.map(partial(stem_comment, stemmer), comments))
    return ret


def make_labels(data) -> np.ndarray:
    """
    :param data: the ``training_data``, ``validation_data``, or ``test_data``.
    :return: list of (multiclass) labels, of shape (N, 6)
    """
    fields = ['toxic', 'severe_toxic', 'obscene', 'threat',
              'insult', 'identity_hate']

    def field2list(f): return getattr(data, f).tolist()

    return np.array(list(zip(*map(field2list, fields))))


def loaddata_train_valid_test():
    trainvalid_data = sklearn.utils.shuffle(get_train_data())
    split = len(trainvalid_data)//2
    validation_data = trainvalid_data[split:]
    training_data = trainvalid_data[:split]
    test_comments, test_labels = get_test_data()
    test_data = test_comments.set_index('id').join(
        other=test_labels.set_index('id'))
    test_data = test_data[test_data.toxic != -1]

    rct_train = training_data.comment_text.tolist()  # (r)aw (c)omment (t)ext
    boc_train = preprocess_comments(rct_train)  # (b)ag (o)f (c)omments
    labels_train = make_labels(training_data)

    rct_valid = validation_data.comment_text.tolist()
    boc_valid = preprocess_comments(rct_valid)
    labels_valid = make_labels(validation_data)

    rct_test = test_data.comment_text.tolist()
    boc_test = preprocess_comments(rct_test)
    labels_test = make_labels(test_data)
    return ((rct_train, boc_train, labels_train),
            (rct_valid, boc_valid, labels_valid),
            (rct_test, boc_test, labels_test))


def vectorize_boc(bag_of_comments, vectorizer=None):
    """
    :param bag_of_comments: lists of preprocessed documents
    :return: a sparse feature matrix of shape (N, M) where N is the length
             of ``bag_of_comments`` and M is the dimension of each feature
             vector of a particular document; if vectorizer is None, the
             trained vectorizer is returned as the second item
    """

    if vectorizer is None:
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
        return (vectorizer.fit_transform(raw_documents=bag_of_comments),
                vectorizer)
    else:
        return vectorizer.transform(raw_documents=bag_of_comments)


def nclass2binOR(labels):
    """
    Make multiclass binary labels to binary classification labels. The binary
    classification predicts True if at least one of the class predicts True.

    Note: not strictly *multiclass*, since the task is to choose k from M
    where k is in range [0, M], rather than to choose 1 from M, where M is the
    number of classes.

    :param labels: the binary label matrix of shape (N, M)
    :return: the label vector for binary classification, of shape (N,)
    """
    return np.sum(labels, axis=1) > 0


def modelspec2modelobj(cls, params):
    args, kwargs = params
    return cls(*args, **kwargs)


def model_eval(X, y, model):
    predictions = (model.predict(X) >= 0.5)
    return sklearn.metrics.fbeta_score(y_true=y, y_pred=predictions, beta=1.5)


def train_and_eval(X_train, y_train, X_valid, y_valid, model) -> float:
    model.fit(X_train, y_train)
    fbeta_train = model_eval(X_train, y_train, model)
    fbeta_valid = model_eval(X_valid, y_valid, model)
    logging.info('{} --- {}'.format(' '.join(
        map(str.strip, str(model).split('\n'))),
        (fbeta_train, fbeta_valid)))
    return fbeta_train, fbeta_valid


def run_models(te, models):
    """
    :param te: the ``train_and_eval`` function with the first four arguments
           filled out
    """
    results = {}
    with multiprocessing.Pool() as pool:
        for ModelClass, all_params in tqdm(models.items(), ascii=True):
            mobjs = map(partial(modelspec2modelobj, ModelClass), all_params)
            fbs = pool.map(te, mobjs)
            results[ModelClass.__name__] = list(zip(fbs, all_params))
    return results


def reduce_dimension(X, k=None, VT=None):
    """
    Reduce dimension of data matrix by sparse matrix SVD.

    :param X: the data matrix
    :param k: the target dimension; ignored if VT is provided
    :param VT: if provided, project X to lower dimension using V; otherwise,
           do SVD and compute the VT
    :return: the reduced dimension matrix, and V if it's not provided
    """
    if VT is None:
        U, S, VT = scipy.sparse.linalg.svds(X, k)
        return np.matmul(U, np.diag(S)), VT
    else:
        return X.dot(VT.T)


models = {
    sklearn.svm.SVC: [
        ((C,), {'kernel': 'linear', 'gamma': 'auto', 'class_weight': 'balanced'})
        for C in np.linspace(0.1, 4.0, 40)
    ],
}


if __name__ == '__main__':
    Xy_cachefile = 'out/Xy_train_valid_bin.pkl'
    try:
        with open(Xy_cachefile, 'rb') as infile:
            X_train, y_train, X_valid, y_valid = pickle.load(infile)
    except (IOError, FileNotFoundError):
        ((_, boc_train, labels_train),
         (_, boc_valid, labels_valid),
         _) = loaddata_train_valid_test()
        X_train, vectorizer = vectorize_boc(boc_train)
        X_valid = vectorize_boc(boc_valid, vectorizer=vectorizer)
        y_train = nclass2binOR(labels_train)
        y_valid = nclass2binOR(labels_valid)
        with open(Xy_cachefile, 'wb') as outfile:
            pickle.dump((X_train, y_train, X_valid, y_valid), outfile)

    logging.info('Xy loaded')

    for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        logging.info('Beginning k={}'.format(k))
        X_train_, VT = reduce_dimension(X_train, k=k)
        X_valid_ = reduce_dimension(X_valid, VT=VT)
        te = partial(train_and_eval, X_train_, y_train, X_valid_, y_valid)
        results = run_models(te, models)
        with open('out/results_dimred-linearsvm_k{}.pkl'.format(k), 'wb') as outfile:
            pickle.dump(results, outfile)
