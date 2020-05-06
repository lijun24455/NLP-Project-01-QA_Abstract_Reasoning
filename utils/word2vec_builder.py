import multiprocessing

from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors

from utils.tools import timeit, load_lines_from_path


def load_data_from_file(path):
    """
    读文本到数组
    :param path: str 文件路径
    :return:字符串数组 ['line_1','line_2'...]
    """
    print('[load_data_from_file] ... <--- {}'.format(path))
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    print('[load_data_from_file] FINISHED! data.len:{} '.format(len(lines)))
    return lines


def save_data_to_file(data, path):
    """
    写数据到文本文件
    :param data:字符串数组
    :param path:文件路径
    """
    print('[save_data_to_file] ... data.len={}'.format(len(data)))
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write('{}\n'.format(line))
    print('[save_data_to_file] FINISHED! ---> {}'.format(path))


@timeit
def build_word2vec(sentens_path, w2v_path, w2v_bin_path, min_count=10, window=5, size=256, sg=1, iter=5):
    print('[build_word2vec] STARTED...')
    print('Train Model STARTED...')
    w2v = Word2Vec(sg=sg, sentences=LineSentence(sentens_path), size=size, window=window, min_count=min_count,
                   workers=multiprocessing.cpu_count(), iter=iter)

    print('Train Model FINISHED! \n Save Model STARTED!')
    w2v.wv.save_word2vec_format(w2v_path, binary=False)
    w2v.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print('Save Model FINISHED!...')


@timeit
def build_word2vec_fast_text(sentens_path, w2v_path, w2v_bin_path, min_count=10, window=5, size=256, iter=5):
    print('[build_word2vec_fast_text] STARTED...')
    print('Train Model STARTED...')
    ft = FastText(sentences=LineSentence(sentens_path), size=size, window=window, min_count=min_count,
                  workers=multiprocessing.cpu_count(), iter=iter)

    print('Train Model FINISHED! \n Save Model STARTED!')
    ft.wv.save_word2vec_format(w2v_path, binary=False)
    ft.wv.save_word2vec_format(w2v_bin_path, binary=True)
    print('Save Model FINISHED!...')


def model_test(model, kw_1, kw_2):
    print('{} vs {} similarity ：{}'.format(kw_1, kw_2, model.wv.similarity(kw_1, kw_2)))
    print('{} similar verbs contains：{}'.format(kw_1, model.wv.similar_by_word(kw_1)))
    print('{} similar verbs contains：{}'.format(kw_2, model.wv.similar_by_word(kw_2)))


# 根据字典构建embedding_matrix
@timeit
def embedding_vector_matrix_build(model, word2idx):
    matrix = {}
    for word in model.vocab:
        if word not in word2idx.keys():
            continue
        matrix[word2idx[word]] = model.get_vector(word)
    return matrix


if __name__ == '__main__':
    train_x_cut_file_path = '../resource/gen/train_x_cut.txt'
    train_y_cut_file_path = '../resource/gen/train_y_cut.txt'
    test_x_cut_file_path = '../resource/gen/test_x_cut.txt'
    vocab_path = '../resource/gen/vocab.txt'

    # gen file path
    all_cut_lines_file_path = '../resource/gen/all_cut_file_path.txt'

    word2vec_file_path = '../resource/gen/word2vec.txt'
    word2vec_bin_file_path = '../resource/gen/word2vec_bin.txt'

    word2vec_ft_file_path = '../resource/gen/word2vec_ft.txt'
    word2vec_ft_bin_file_path = '../resource/gen/word2vec_ft_bin.txt'

    # 训练开关，缺省关闭
    train_switch = False

    # 训练词向量
    if train_switch:
        all_cut_lines = []
        all_cut_lines += load_data_from_file(train_x_cut_file_path)
        all_cut_lines += load_data_from_file(train_y_cut_file_path)
        all_cut_lines += load_data_from_file(test_x_cut_file_path)

        save_data_to_file(all_cut_lines, all_cut_lines_file_path)

        build_word2vec(all_cut_lines_file_path, word2vec_file_path, word2vec_bin_file_path,
                       min_count=10, window=5, size=256, sg=1, iter=3)

        build_word2vec_fast_text(all_cut_lines_file_path, word2vec_ft_file_path, word2vec_ft_bin_file_path,
                                 min_count=10,
                                 window=10, size=256, iter=3)

    # 测试1 : 检查下词向量效果
    # 加载w2v模型
    w2v_model = KeyedVectors.load_word2vec_format(word2vec_bin_file_path, binary=True)
    model_test(w2v_model, '宝马', '奔驰')
    model_test(w2v_model, '汽车', '减速')
    model_test(w2v_model, '汽车', '车')
    model_test(w2v_model, '技师', '车主')
    model_test(w2v_model, '火花塞', '减震器')
    model_test(w2v_model, '刹车片', '解答')

    # 加载fast_text模型
    ft_model = KeyedVectors.load_word2vec_format(word2vec_ft_bin_file_path, binary=True)
    model_test(ft_model, '宝马', '奔驰')
    model_test(ft_model, '汽车', '减速')
    model_test(ft_model, '汽车', '车')
    model_test(ft_model, '技师', '车主')
    model_test(ft_model, '火花塞', '减震器')
    model_test(ft_model, '刹车片', '解答')

    # hw-2 作业：创建embedding_matrix
    word2idx = {}
    idx2word = {}
    vocab_list = load_lines_from_path(vocab_path)

    for line in vocab_list:
        segments = line.split()
        word2idx[segments[0]] = segments[1]
        idx2word[segments[1]] = segments[0]

    # 创建embedding_matrix
    w2v_embedding_matrix = embedding_vector_matrix_build(w2v_model, word2idx)
    ft_embedding_matrix = embedding_vector_matrix_build(ft_model, word2idx)

    print('w2v matrix size:{}'.format(len(w2v_embedding_matrix)))
    print('ft matrix size:{}'.format(len(ft_embedding_matrix)))

    # 测试2：:检查下embedding_matrix有没错
    t_word = '奔驰'
    t_idx = word2idx[t_word]
    print('tWord:{}, tIdx:{}'.format(t_word, t_idx))
    wv_from_w2v_matrix = w2v_embedding_matrix[t_idx]
    wv_from_w2v_model = w2v_model.get_vector(t_word)
    print('wv_from_w2v_matrix == wv_from_w2v_model?:{}'.format((wv_from_w2v_matrix == wv_from_w2v_model)))

    wv_from_ft_matrix = ft_embedding_matrix[t_idx]
    wv_from_ft_model = ft_model.get_vector(t_word)
    print('wv_from_ft_matrix == wv_from_ft_matrix?:{}'.format((wv_from_ft_matrix == wv_from_ft_model)))
