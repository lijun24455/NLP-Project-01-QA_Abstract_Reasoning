from utils.tools import Vocab

if __name__ == '__main__':
    word2vec_file_path = '../resource/gen/word2vec.txt'
    word2vec_bin_file_path = '../resource/gen/word2vec_bin.txt'

    word2vec_ft_file_path = '../resource/gen/word2vec_ft.txt'
    word2vec_ft_bin_file_path = '../resource/gen/word2vec_ft_bin.txt'

    vocabs_file_path = '../resource/gen/vocabs_w_f.txt'

    vocab_helper = Vocab(vocabs_file_path)
    print(vocab_helper.size())
    print(vocab_helper.get_id_by_word('奔驰'))
    print(vocab_helper.get_id_by_word('没有的词'))
    print(vocab_helper.get_word_by_id(1))
