# Data Preprocess
# 对数据做预处理，去除缺失数据、去除噪音词汇和符号；使用jieba进行中文分词处理
# @author lijun
import pandas as pd
import jieba


class Preprocessor:

    def get_file_path(self, file_name):
        return '../resource/' + file_name

    def load_text_from_cvs(self, train_file_name):
        print('loading file {} ...'.format(train_file_name))
        df = pd.read_csv(self.get_file_path(train_file_name), encoding='utf=8', nrows=3900)
        rows = len(df)
        print('file {} has {} rows.'.format(train_cvs_file_name, rows))
        text = ''
        i = 0
        for i in range(rows):
            print('current line : {}'.format(i))
            # if (i % 10000) == 0:
            #     print('current line : {}'.format(i))
            row = df.iloc[i]
            line_text = ' '.join(row[3:])
            line_text.strip()
            text += line_text
        print('loading file finished ! ')
        return text

    def get_verb_set_from_text(self, text):
        print('get verbs from text ... ')
        verb_set = set()
        verbs = jieba.cut(text)
        for verb in verbs:
            verb_set.add(verb)
        print('get verbs from text finished! ')
        print('verb set size : {}'.format(len(verb_set)))
        return verb_set

    def save_verb_set_to_file(self, verb_set, filename):
        file_path = self.get_file_path(filename)
        print('write verbs to file : {} ...'.format(file_path))
        i = 1
        with open(file_path, 'w') as f:
            for verb in verb_set:
                if i == 1:
                    f.write('{} {}'.format(verb, i))
                else:
                    f.write('\n{} {}'.format(verb, i))
                i = i + 1
        print('write verbs to file : {} finished!'.format(file_path))

if __name__ == '__main__':
    # 82943 rows
    train_cvs_file_name = 'AutoMaster_TrainSet.csv'
    verb_file_name = 'vocab.txt'

    processor = Preprocessor()
    text = processor.load_text_from_cvs(train_cvs_file_name)
    v_set = processor.get_verb_set_from_text(text)
    processor.save_verb_set_to_file(v_set, verb_file_name)
