import tensorflow as tf
import time
import numpy as np

from batcher import batcher
from loss_function import loss_function
from utils.config import *


def get_train_msg():
    # 获得已训练的轮次
    path = os.path.join(SEQ2SEQ_CKPT, "trained_epoch.txt")
    with open(path, mode="r", encoding="utf-8") as f:
        trained_epoch = int(f.read())
    return trained_epoch

ß
def save_train_msg(trained_epoch):
    # 保存训练信息（已训练的轮数）
    path = os.path.join(SEQ2SEQ_CKPT, "trained_epoch.txt")
    with open(path, mode="w", encoding="utf-8") as f:
        f.write(str(trained_epoch))


def train_model(model, vocab, params, checkpoint_manager):
    print(vocab)

    epochs = params['epochs']
    batch_size = params['batch_size']

    start_index = vocab.get_id_by_word('[START]')
    pad_index = vocab.get_id_by_word('[PAD]')
    print('train_model:start_index:{}, pad_index:{}'.format(start_index, pad_index))

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=params["learning_rate"])

    def train_step(enc_input, dec_target):
        # dec_target [4980, 939, 41, 27, 4013, 815, 14702]

        with tf.GradientTape() as tape:
            # enc_output (batch_size, enc_len, enc_unit)
            # enc_hidden (batch_size, enc_unit)
            enc_output, enc_hidden = model.encoder(enc_input)

            # 第一个decoder输入 开始标签
            # dec_input (batch_size, 1)
            dec_input = tf.expand_dims([start_index] * batch_size, 1)

            # 第一个隐藏层输入
            # dec_hidden (batch_size, enc_unit)
            dec_hidden = enc_hidden
            # 逐个预测序列
            # predictions (batch_size, dec_len-1, vocab_size)
            predictions, _ = model(dec_input, dec_hidden, enc_output, dec_target)

            _batch_loss = loss_function(dec_target[:, 1:], predictions)

        variables = model.trainable_variables
        gradients = tape.gradient(_batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return _batch_loss

    dataset = batcher(vocab, params)
    steps_per_epoch = params["steps_per_epoch"]

    for epoch in range(params['epochs']):
        start = time.time()
        step = 0
        total_loss = 0
        # print(len(dataset.take(params['steps_per_epoch'])))
        for (step, batch) in enumerate(dataset.take(params['steps_per_epoch'])):
            # 讲设你的样本数是1000，batch size10,一个epoch，我们一共有100次，200， 500， 40，20.
            # batch_loss = train_step(batch[0]["enc_input"],  # shape=(16, 200)
            #                         batch[1]["dec_target"])  # shape=(16, 50)

            inputs = batch["enc_input"]
            target = batch["target"]

            batch_loss = train_step(inputs, target)

            total_loss += batch_loss

            # step += 1
            # if step % 100 == 0:
            #     print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, batch_loss.numpy()))
            if (step + 1) % 1 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(params["trained_epoch"] + epoch + 1,
                                                             step + 1,
                                                             batch_loss.numpy())
                      )

            if params["debug_mode"]:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             step,
                                                             batch_loss.numpy()))
                if step >= 10:
                    break

        if epoch % 1 == 0:  # 改成增加评价函数，在验证集上做验证，只要比上次好就保存一次
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path,
                                                                              total_loss / step))
            try:
                record_file = os.path.join(SEQ2SEQ_CKPT, "record.txt")
                with open(record_file, mode="w", encoding="utf-8") as f:
                    f.write('Epoch {} Loss {:.4f}\n'.format(params["trained_epoch"] + epoch + 1,
                                                            total_loss / steps_per_epoch))
            except:
                pass
            print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            # 学习率的衰减，按照训练的次数来更新学习率（tf1.x） 加快收敛，一开始可以快速下降，越到后越小步以达到最低；
            lr = params['learning_rate'] * np.power(0.9, epoch + 1)
            optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=lr)
            print("learning_rate=", optimizer.get_config()["learning_rate"])
            save_train_msg(params["trained_epoch"] + epoch + 1)  # 保存已训练的轮数

        print('Epoch {} Loss {:.4f}'.format(params["trained_epoch"] + epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
