import argparse

import os
import pathlib

# 一些变量
from utils.tools import *
from utils.vocab import Vocab

root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 训练数据与测试数据集

TRAIN_DATA = os.path.join(root, 'resource', 'AutoMaster_TrainSet.csv')
TEST_DATA = os.path.join(root, 'resource', 'AutoMaster_TestSet.csv')
STOP_WORDS = os.path.join(root, 'resource', 'stopwords', 'i_stop_words.txt')

# 预处理过程中生成的数据
RAW_TEXT = os.path.join(root, 'data', 'raw_text.txt')  # 原始文本
PROC_TEXT = os.path.join(root, 'data', 'proc_text.txt')  # 预处理后的文本
USER_DICT = os.path.join(root, 'data', 'user_dict_new.txt')  # 自定义词典
#
TRAIN_SEG = os.path.join(root, 'data', 'train_seg.csv')  # 预处理后的csv文件
TEST_SEG = os.path.join(root, 'data', 'test_seg.csv')

# 2. pad oov处理后的数据
TRAIN_X_PAD = os.path.join(root, 'data', 'train_X_pad.csv')
TRAIN_Y_PAD = os.path.join(root, 'data', 'train_Y_pad.csv')
TEST_X_PAD = os.path.join(root, 'data', 'test_X_pad.csv')

# 训练数据
TRAIN_SEG_X = os.path.join(root, 'resource', 'gen', 'train_x_cut.txt')
TRAIN_SEG_Y = os.path.join(root, 'resource', 'gen', 'train_y_cut.txt')
TEST_SEG_X = os.path.join(root, 'resource', 'gen', 'test_x_cut.txt')
DATASET_MSG = os.path.join(root, 'resource', 'gen', 'dataset_msg.txt')

TRAIN_X = os.path.join(root, 'resource', 'gen', 'train_x_cut.txt')
TRAIN_Y = os.path.join(root, 'resource', 'gen' 'train_y_cut.txt')
TEST_X = os.path.join(root, 'resource', 'gen' 'test_x_cut.txt')

# 词向量模型
WV_MODEL = os.path.join(root, 'resource', 'gen', 'word2vec_ft_bin')
VOCAB = os.path.join(root, 'resource', 'gen', 'vocabs_w_f.txt')
EMBEDDING_MATRIX = os.path.join(root, 'resource', 'gen', 'word_embedding')
VOCAB_PAD = os.path.join(root, 'data', 'wv', 'vocab_index_pad.txt')
EMBEDDING_MATRIX_PAD = os.path.join(root, 'data', 'wv', 'embedding_matrix_pad.txt')
WV_MODEL_PAD = os.path.join(root, 'data', 'wv', 'word2vec_pad.model')

# 存档
SEQ2SEQ_CKPT = os.path.join(root, 'resource', 'model', 'seq2seq', 'checkpoints')
PGN_CKPT = os.path.join(root, 'resource', 'model', 'pgn', 'checkpoints')
# 其他
FONT = os.path.join(root, 'data', 'TrueType', 'simhei.ttf')
PARAMS_FROM_DATASET = os.path.join(root, 'data', 'params_from_dataset.txt')

# 结果
RESULT_DIR = os.path.join(root, 'resource', 'result')

TRAIN_PICKLE_DIR = os.path.join(root, 'resource', 'dataset')

EPOCH = 4
BATCH_SIZE = 16
NUM_SAMPLES = 81391


def get_result_file_name():
    """
    获取结果文件的名称
    :return:  20191209_16h49m32s_res.csv
    """
    now = time.strftime('%Y_%m_%d_%H_%M_%S')

    file_name = os.path.join(RESULT_DIR, now + "_result.csv")
    return file_name


def get_params():
    # vocab = Vocab(VOCAB_PAD)
    vocab = Vocab(VOCAB)
    steps_per_epoch = NUM_SAMPLES // BATCH_SIZE  # 不算多余的
    parser = argparse.ArgumentParser()

    # 调试选项
    parser.add_argument("--mode", default='train', help="run mode", type=str)
    parser.add_argument("--decode_mode", default='greedy', help="decode mode greedy/beam", type=str)
    parser.add_argument("--greedy_decode", default=True, help="if greedy decode", type=bool)
    parser.add_argument("--debug_mode", default=True, help="debug mode", type=bool)
    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--pointer_gen", default=True,
                        help="if use pointer generator model",
                        type=bool)
    parser.add_argument("--cov_loss_wt", default=0.5, help="coverage loss weight", type=float)

    # 预处理后的参数
    parser.add_argument("--max_enc_len",
                        default=200,
                        help="Encoder input max sequence length",
                        type=int)
    parser.add_argument("--max_dec_len",
                        default=40,
                        help="Decoder input max sequence length",
                        type=int)
    parser.add_argument("--vocab_size", default=vocab.count, help="max vocab size , None-> Max ", type=int)

    # 训练参数设置
    parser.add_argument("--batch_size", default=BATCH_SIZE, help="batch size", type=int)
    parser.add_argument("--epochs", default=EPOCH, help="train epochs", type=int)
    parser.add_argument("--steps_per_epoch", default=steps_per_epoch, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_steps", default=2, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--trained_epoch", default=0, help="trained epoch", type=int)

    # 优化器
    parser.add_argument("--learning_rate", default=0.01, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)

    # 模型参数
    parser.add_argument("--embed_size",
                        default=256,
                        help="Words embeddings dimension",
                        type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256, help="[context vector, decoder state, decoder input] feedforward \
                            result dimension - this result is used to compute the attention weights",
                        type=int)

    # 相关文件路径
    parser.add_argument("--vocab_path", default=VOCAB, help="vocab path", type=str)
    parser.add_argument("--w2v_output", default=WV_MODEL, help="w2v_bin path", type=str)
    parser.add_argument("--train_seg_x_dir", default=TRAIN_SEG_X, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=TRAIN_SEG_Y, help="train_seg_y_dir", type=str)
    parser.add_argument("--test_seg_x_dir", default=TEST_SEG_X, help="train_seg_x_dir", type=str)

    parser.add_argument("--result_save_path", default=get_result_file_name(),
                        help='result save path', type=str)

    # 暂时不确定有何用
    # parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)

    args = parser.parse_args()
    _params = vars(args)

    return _params


""" 使用方法：
新建一个文件xx.py
from utils.params import get_params
params = get_params()

在jupyter notebook里
%run xx.py
就能得到以 params 为名的字典
"""

if __name__ == "__main__":
    # get_params_from_dataset(check=True)
    params = get_params()
    print('vocab_path:',params['vocab_path'])
