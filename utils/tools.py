import os
import pickle
import time
import re
import numpy as np

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


class Vocab:
    def __init__(self, vocab_file_path, vocab_max_size=None):
        self.PAD_TOKEN = '[PAD]'
        self.UNKNOWN_TOKEN = '[UNK]'
        self.START_DECODING = '[START]'
        self.STOP_DECODING = '[STOP]'

        self.MASK = ['[PAD]', '[UNK]', '[START]', '[STOP]']
        self.MASK_LEN = len(self.MASK)
        self.pad_token_idx = self.MASK.index(self.PAD_TOKEN)
        self.unk_token_idx = self.MASK.index(self.UNKNOWN_TOKEN)
        self.start_token_idx = self.MASK.index(self.START_DECODING)
        self.stop_token_idx = self.MASK.index(self.STOP_DECODING)

        self.word2id, self.id2word = self.load_vocab(vocab_file_path, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, vocab_file_path, vocab_max_size):
        word2id = {mask: idx for idx, mask in enumerate(self.MASK)}
        id2word = {idx: mask for idx, mask in enumerate(self.MASK)}

        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print('file lines cnt : {}'.format(len(lines)))
        for line in lines:
            word, id = line.strip().split()
            id = int(id)
            if vocab_max_size and id > vocab_max_size - self.MASK_LEN - 1:
                break
            word2id[word] = id + self.MASK_LEN
            id2word[id + self.MASK_LEN] = word
        return word2id, id2word

    def get_id_by_word(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        else:
            return self.word2id[word]

    def get_word_by_id(self, id):
        if id not in self.id2word:
            return self.id2word[self.unk_token_idx]
        else:
            return self.id2word[id]

    def size(self):
        return self.count

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