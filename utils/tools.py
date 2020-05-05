import time
import re

from gensim.models import KeyedVectors


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


def save_lines_to_path(lines, path):
    print('[save_lines_to_path] lines.len:{}, path:{}'.format(len(lines), path))
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write('{}\n'.format(line))

# 根据字典构建embedding_matrix
@timeit
def embedding_vector_matrix_build(model, word2idx):
    matrix = {}
    for word, idx in word2idx.items():
        if word in model:
            matrix[idx] = model.get_vector(word)
    return matrix


if __name__ == '__main__':
    vocab_path = '../resource/gen/vocab.txt'
    w2v_bin_path = '../resource/gen/word2vec_bin.txt'
    ft_bin_path = '../resource/gen/word2vec_ft_bin.txt'

    word2idx = {}
    idx2word = {}
    vocab_list = load_lines_from_path(vocab_path)

    for line in vocab_list:
        segments = line.split()
        word2idx[segments[0]] = segments[1]
        idx2word[segments[1]] = segments[0]

    # 加载模型
    w2v_model = KeyedVectors.load_word2vec_format(w2v_bin_path, binary=True)
    ft_model = KeyedVectors.load_word2vec_format(ft_bin_path, binary=True)

    w2v_embedding_matrix = embedding_vector_matrix_build(w2v_model, word2idx)
    ft_embedding_matrix = embedding_vector_matrix_build(ft_model, word2idx)

    print('w2v matrix size:{}'.format(len(w2v_embedding_matrix)))
    print('ft matrix size:{}'.format(len(ft_embedding_matrix)))

    # 测试
    t_word = '奔驰'
    t_idx = word2idx[t_word]
    print('tWord:{}, tIdx:{}'.format(t_word, t_idx))
    wv_from_w2v_matrix = w2v_embedding_matrix[t_idx]
    wv_from_w2v_model = w2v_model.get_vector(t_word)
    print('wv_from_w2v_matrix == wv_from_w2v_model?:{}'.format((wv_from_w2v_matrix == wv_from_w2v_model)))

    wv_from_ft_matrix = ft_embedding_matrix[t_idx]
    wv_from_ft_model = ft_model.get_vector(t_word)
    print('wv_from_ft_matrix == wv_from_ft_matrix?:{}'.format((wv_from_ft_matrix == wv_from_ft_model)))
