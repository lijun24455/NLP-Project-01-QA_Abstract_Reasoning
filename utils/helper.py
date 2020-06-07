import os
import pathlib
import sys
from collections import defaultdict
import tensorflow as tf


def save_file_segment(from_path, to_path, line_cnt):
    assert line_cnt > 0
    lines = []
    with open(from_path, 'r', encoding='utf-8') as f:
        for _ in range(line_cnt):
            lines.append(f.readline())
    with open(to_path, 'w', encoding='utf-8')as f:
        f.writelines(lines)


if __name__ == '__main__':
    train_x_path = '../resource/gen/train_x_cut.txt'
    train_y_path = '../resource/gen/train_y_cut.txt'
    test_x_path = '../resource/gen/test_x_cut.txt'

    # gen files
    train_x_seg_path = '../resource/demo/train_x_cut.txt'
    train_y_seg_path = '../resource/demo/train_y_cut.txt'
    test_x_seg_path = '../resource/demo/test_x_cut.txt'

    # save_file_segment(train_x_path, train_x_seg_path, 20)
    # save_file_segment(train_y_path, train_y_seg_path, 20)
    # save_file_segment(test_x_path, test_x_seg_path, 20)

    # seq_cnt_dict = defaultdict(int)
    # with open(train_x_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         seq_len = len(line.split())
    #         seq_cnt_dict[seq_len] += 1
    #
    # seq_cnt_dict = sorted(seq_cnt_dict.items(), key=lambda d: d[1], reverse=True)
    # print(len(seq_cnt_dict))
    # print( seq_cnt_dict)

    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # sys.path.append(BASE_DIR)
    # print(BASE_DIR)
    #
    # root = pathlib.Path(os.path.abspath(__file__)).parent.parent
    # print(root)

    vocab_path = "../resource/gen/vocabs_w_f.txt"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.split()) < 2:
                print(line)
                print(len(line), '------>', len(line.strip()))

