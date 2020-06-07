import os
import pickle
import time
import re
import numpy as np
from gensim.models import KeyedVectors

from utils.config import *


def timeit(f):
    def wrapper(*args, **kwargs):
        print('[{}] 开始!'.format(f.__name__))
        start_time = time.time()
        res = f(*args, **kwargs)
        end_time = time.time()
        print("[%s] 完成! -> 运行时间为：%.8f" % (f.__name__, end_time - start_time))
        return res

    return wrapper


# 去掉多余空格
def clean_space(text):
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list, key=lambda i: len(i), reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i, new_i)
    return text


def load_lines_from_path(path, no_space=False):
    print('[load_lines_from_path] <-- {}'.format(path))
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if no_space:
                line = clean_space(line).strip()
            else:
                line = line.strip()
            lines.append(line)
    return lines


@timeit
def save_dict_to_path(dic, path):
    dic_size = len(dic)
    print('[save_dict_to_path] 写入字典大小：' + dic_size)
    with open(path, 'w', encoding='utf-8') as f:
        for k, v in dic.items():
            f.write('{} {}\n'.format(k, v))
            i = i + 1


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
        print("save %s ok." % pkl_path)


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def save_lines_to_path(lines, path):
    print('[save_lines_to_path] lines.len:{}, path:{}'.format(len(lines), path))
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('{}\n'.format(line))


def load_train_dataset():
    """
    :return: 加载处理好的数据集
    """
    train_x = np.loadtxt(TRAIN_X, delimiter=",", dtype=np.float32)
    train_y = np.loadtxt(TRAIN_Y, delimiter=",", dtype=np.float32)

    return train_x, train_y


def load_test_dataset():
    """
    :return: 加载处理好的数据集
    """
    test_x = np.loadtxt(TEST_X, delimiter=",", dtype=np.float32)
    return test_x


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


def load_embedding_matrix(w2v_model_path, vocab_path, vocab_size, embed_size):
    """
    load pretrain word2vec weight matrix
    :param vocab_size:
    :return embedding_matrix:
    """
    # word2vec_dict = load_pkl(params['word2vec_output'])
    w2v = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
    vocab_dict = open(vocab_path, encoding='utf-8').readlines()
    print('[load_word2vec]:vocab_dict.len:{}'.format(len(vocab_dict)))
    embedding_matrix = np.zeros((vocab_size, embed_size))

    for line in vocab_dict[:vocab_size]:
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



