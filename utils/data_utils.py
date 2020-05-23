import numpy as np
import pickle
import os
import copy
from collections import Counter

from gensim.models import KeyedVectors


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def load_word2vec(params):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return embedding_matrix:
    """
    # word2vec_dict = load_pkl(params['word2vec_output'])
    w2v = KeyedVectors.load_word2vec_format(params['word2vec_output'], binary=True)
    vocab_dict = open(params['vocab_path'], encoding='utf-8').readlines()
    print('[load_word2vec]:vocab_dict.len:{}'.format(len(vocab_dict)))
    embedding_matrix = np.zeros((params['vocab_size'], params['embed_size']))

    for line in vocab_dict[:params['vocab_size']]:
        word_id = line.split()
        if len(word_id) < 2:
            print('empty word:{}'.format(line))
            continue
        word, i = word_id
        if word in w2v.vocab:
            embedding_matrix[int(i)] = w2v[word]
        else:
            embedding_matrix[int(i)] = np.random.uniform(-10, 10, 256)
    print('embedding_m.shape:{}'.format(embedding_matrix.shape))
    return embedding_matrix


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)
