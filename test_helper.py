import tensorflow as tf
from tqdm import tqdm


def greedy_decode(model, test_x, vocab, params):
    # 存储结果
    batch_size = params["batch_size"]
    results = []

    sample_size = len(test_x)
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = sample_size // batch_size + 1
    # [0,steps_epoch)
    for i in tqdm(range(steps_epoch)):
        batch_data = test_x[i * batch_size:(i + 1) * batch_size]
        results += batch_greedy_decode(model, batch_data, vocab, params)
    return results


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    # print(enc_data)
    global outputs
    batch_data = enc_data[0]["enc_input"]
    batch_size = enc_data[0]["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    inputs = batch_data

    enc_output, enc_hidden = model.encoder(inputs)
    dec_hidden = enc_hidden
    # 这里解释下为什么要有一个batch_size,因为训练得时候是按照一个batch size扔进去得，所以得到得模型得输入结构也是如此，因此在测试得时候相当于将单个样本
    # 乘以batch size那么多遍，然后再得到结果，结果区list得第一个即可，当然理论上list得内容是一样得
    dec_input = tf.constant([vocab.get_id_by_word('[START]')] * batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)
    # print('enc_output shape is :',enc_output.get_shape())
    # print('dec_hidden shape is :', dec_hidden.get_shape())
    # print('inputs shape is :', inputs.get_shape())
    # print('dec_input shape is :', dec_input.get_shape())
    context_vector, _ = model.attention(dec_hidden, enc_output)

    for t in range(params['max_dec_len']):
        # 单步预测
        # final_dist (batch_size, 1, vocab_size+batch_oov_len)
        predictions, dec_hidden = model.decoder(dec_input,
                                                dec_hidden,
                                                enc_output,
                                                context_vector)

        # id转换
        predicted_ids = tf.argmax(predictions, axis=1).numpy()
        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.get_word_by_id(predicted_id) + ' '
        # dec_input = tf.expand_dims(predicted_ids, 1)
    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断
        if '[STOP]' in predict:
            # 截断stop
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    return results
