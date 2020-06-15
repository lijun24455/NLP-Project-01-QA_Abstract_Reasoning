import numpy as np

from utils.vocab import Vocab


def padding(sentence, max_len, vocab):
    """
    给句子加上<START><PAD><UNK><END>
    :param sentence:
    :param max_len:
    :param vocab:
    :return:
    """
    words = sentence.strip().split()
    words = words[:max_len]
    sentence = [word if word in vocab.word2id else vocab.UNKNOWN_TOKEN for word in words]

    sentence = [vocab.START_DECODING] + sentence + [vocab.STOP_DECODING]
    sentence = sentence + [vocab.PAD_TOKEN] * (max_len - len(words))
    return ' '.join(sentence)


def transform_to_ids(sentence, vocab):
    # 字符串切分成词
    words = sentence.split()
    # 按照vocab的index进行转换
    # ids = [_vocab[word] if word in _vocab else _vocab['<UNK>'] for word in words]
    ids = [vocab.get_id_by_word(word) for word in words]
    return ids


def seg_text_to_ids(seg_text_path, vocab, max_len):
    print('开始数据进行预处理...')
    with open(seg_text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        line = padding(line, max_len, vocab)
        line = transform_to_ids(line, vocab)
        result.append(line)

    return result


def save_dataset_file(dataset, path):
    np.savetxt(path, dataset, fmt="%d", delimiter=",")


if __name__ == '__main__':
    test_seg_text_path = '../resource/gen/test_x_cut.txt'
    test_dataset_path = '../resource/gen/test_x_dataset.txt'
    vocab_file = '../resource/gen/vocabs_w_f.txt'
    max_len = 200

    vocab = Vocab(vocab_file, vocab_max_size=50000)
    dataset = seg_text_to_ids(test_seg_text_path, vocab, max_len)
    save_dataset_file(dataset, test_dataset_path)