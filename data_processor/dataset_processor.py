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


def seg_text_to_ids(seg_text_path, vocab, max_len):
    print('开始数据进行预处理...')
    

    pass


if __name__ == '__main__':
    for i in range(1, 11):
        print(i)
