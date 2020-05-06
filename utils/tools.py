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
