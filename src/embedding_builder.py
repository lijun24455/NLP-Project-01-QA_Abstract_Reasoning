import numpy as np
from gensim.models import KeyedVectors

from src.utils.tools import *

if __name__ == '__main__':
    word2vec_file_path = '../resource/gen/word2vec.txt'
    word2vec_bin_file_path = '../resource/gen/word2vec_bin.txt'

    word2vec_ft_file_path = '../resource/gen/word2vec_ft.txt'
    word2vec_ft_bin_file_path = '../resource/gen/word2vec_ft_bin.txt'

    vocabs_file_path = '../resource/gen/vocabs_w_f.txt'

    embedding_output_path = '../resource/gen/word_embedding'

    vocab_helper = Vocab(vocabs_file_path, 50000)
    print(vocab_helper.size())
    print(vocab_helper.get_id_by_word('奔驰'))
    print(vocab_helper.get_id_by_word('没有的词'))
    print(vocab_helper.get_word_by_id(1))

    model = KeyedVectors.load_word2vec_format(word2vec_bin_file_path, binary=True)

    word_dict = {}

    for word, index in vocab_helper.word2id.items():
        if word in model.vocab:
            word_dict[index] = model[word]
        else:
            word_dict[index] = np.random.uniform(-10, 10, 256)

    print('字典大小：{}'.format(len(word_dict)))

    # save_dict_to_path(word_dict, embedding_output_path)
    dump_pkl(word_dict, embedding_output_path)