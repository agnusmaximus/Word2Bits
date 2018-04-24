import numpy as np
from scipy.sparse import csr_matrix


def save_matrix(f, m):
    np.savez_compressed(f, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape)


def load_matrix(f):
    if not f.endswith('.npz'):
        f += '.npz'
    loader = np.load(f)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def save_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w in vocab:
            print >>f, w


def load_vocabulary(path):
    with open(path) as f:
        vocab = [line.strip() for line in f if len(line) > 0]
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab


def save_count_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w, c in vocab:
            print >>f, w, c


def load_count_vocabulary(path):
    with open(path) as f:
        # noinspection PyTypeChecker
        vocab = dict([line.strip().split() for line in f if len(line) > 0])
    return vocab
