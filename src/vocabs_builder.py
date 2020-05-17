from collections import defaultdict

from src.utils.tools import timeit

train_x_cut_file_path = '../resource/gen/train_x_cut.txt'
train_y_cut_file_path = '../resource/gen/train_y_cut.txt'
test_x_cut_file_path = '../resource/gen/test_x_cut.txt'

vocabs_w_f_file_path = '../resource/gen/vocabs_w_f.txt'
vocabs_f_w_file_path = '../resource/gen/vocabs_f_w.txt'


# 加载数据
@timeit
def load_data(path):
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            words += line.split()
    return words


@timeit
def build_vocabs(items, sort=True, min_count=0, lower=False):
    """
    构建词典
    :param items: 词典列表
    :param sort: 是否排序（按词频）
    :param min_count: 最小词频
    :param lower: 是否小写
    :return: r_1:(word,freq), r_2(freq, word)
    """
    print('[building vocabs]...')
    result = []
    # defaultdict 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值,int的默认值是0
    dic = defaultdict(int)
    for item in items:
        if lower:
            item.lower()
        dic[item] += 1
    if sort:
        # (w,i)
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            if item[1] > min_count:
                # (w,i)
                result.append(item)
    else:
        for i, item in enumerate(dic.items()):
            if item[1] > min_count:
                # (w,i)
                result.append(item)

    vocabs_w_i = [(w[0], i) for i, w in enumerate(result)]
    vocabs_i_w = [(i, w[0]) for i, w in enumerate(result)]
    print('[building vocabs] FINISHED!')

    return vocabs_w_i, vocabs_i_w


@timeit
def save_vocabs_to_file(vocabs, path):
    print('[save vocabs file]...')
    with open(path, 'w', encoding='utf-8') as f:
        for w, i in vocabs:
            f.write("{} {}\n".format(w, i))
    print('[save vocabs file] FINISHED!...Path:{}'.format(path))


if __name__ == '__main__':
    train_x_words = load_data(train_x_cut_file_path)
    train_y_words = load_data(train_y_cut_file_path)
    test_x_words = load_data(test_x_cut_file_path)
    all_words = train_x_words + train_y_words + train_x_words
    vocabs_w_f, vocabs_f_w = build_vocabs(all_words)

    save_vocabs_to_file(vocabs_w_f, vocabs_w_f_file_path)
    save_vocabs_to_file(vocabs_f_w, vocabs_f_w_file_path)
