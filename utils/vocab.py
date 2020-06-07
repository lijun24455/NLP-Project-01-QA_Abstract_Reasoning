

class Vocab:
    def __init__(self, vocab_file_path, vocab_max_size=None):
        self.PAD_TOKEN = '[PAD]'
        self.UNKNOWN_TOKEN = '[UNK]'
        self.START_DECODING = '[START]'
        self.STOP_DECODING = '[STOP]'

        self.MASK = ['[PAD]', '[UNK]', '[START]', '[STOP]']
        self.MASK_LEN = len(self.MASK)
        self.pad_token_idx = self.MASK.index(self.PAD_TOKEN)
        self.unk_token_idx = self.MASK.index(self.UNKNOWN_TOKEN)
        self.start_token_idx = self.MASK.index(self.START_DECODING)
        self.stop_token_idx = self.MASK.index(self.STOP_DECODING)

        self.word2id, self.id2word = self.load_vocab(vocab_file_path, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, vocab_file_path, vocab_max_size):
        word2id = {mask: idx for idx, mask in enumerate(self.MASK)}
        id2word = {idx: mask for idx, mask in enumerate(self.MASK)}

        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print('file lines cnt : {}'.format(len(lines)))
        for line in lines:
            word, id = line.strip().split()
            id = int(id)
            if vocab_max_size and id > vocab_max_size - self.MASK_LEN - 1:
                break
            word2id[word] = id + self.MASK_LEN
            id2word[id + self.MASK_LEN] = word
        return word2id, id2word

    def get_id_by_word(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        else:
            return self.word2id[word]

    def get_word_by_id(self, id):
        if id not in self.id2word:
            return self.id2word[self.unk_token_idx]
        else:
            return self.id2word[id]

    def size(self):
        return self.count