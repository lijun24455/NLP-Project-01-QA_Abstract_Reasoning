import tensorflow as tf

from seq2seq.model import Seq2Seq
from utils.batcher import beam_test_batch_generator
from utils.test_helper import greedy_decode
import pandas as pd

from utils.config import *
from utils.tools import *


def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    # config_gpu()

    print('[Test]Building the model...')
    model = Seq2Seq(params)
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    ckpt = tf.train.Checkpoint(Seq2Seq=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, SEQ2SEQ_CKPT, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    if params['greedy_decode']:
        params['batch_size'] = 512
        results = predict_result(model, params, vocab)
    else:
        b = beam_test_batch_generator(params['beam_size'])
        results = []
        for batch in b:
            best_hyp = beam_decode(model, batch, vocab, params)
            results.append(best_hyp.abstract)
        save_predict_result(results, params['result_save_path'])
    return results


def predict_result(model, params, vocab):
    test_x = load_test_dataset()
    # 预测结果
    results = greedy_decode(model, test_x, vocab, params)
    results = list(map(lambda x: x.replace(" ", ""), results))
    # 保存结果
    save_predict_result(results, params['result_save_path'])

    return results


def save_predict_result(results, result_save_path):
    # 读取结果
    test_df = pd.read_csv(TEST_DATA)
    # 填充结果
    test_df['Prediction'] = results
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    test_df.to_csv(result_save_path, index=None, sep=',')


# def test(params):
#     global model, ckpt, checkpoint_dir
#     assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
#     # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
#
#     print("Building the model ...")
#     if params["model"] == "SequenceToSequence":
#         model = Seq2Seq(params)
#     print("Creating the vocab ...")
#     vocab = Vocab(params["vocab_path"], params["vocab_size"])
#
#     print("Creating the batcher ...")
#     b = batcher(vocab, params)
#
#     print("Creating the checkpoint manager")
#     if params["model"] == "SequenceToSequence":
#         checkpoint_dir = "{}/checkpoint".format(params["seq2seq_model_dir"])
#         ckpt = tf.train.Checkpoint(step=tf.Variable(0), SequenceToSequence=model)
#     ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
#
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print("Model restored")
#     for batch in b:
#         yield batch_greedy_decode(model, batch, vocab, params)


# def test_and_save(params):
#     assert params["test_save_dir"], "provide a dir where to save the results"
#     gen = test(params)
#     results = []
#     with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
#         for i in range(params["num_to_test"]):
#             trial = next(gen)
#             trial = list(map(lambda x: x.replace(" ", ""), trial))
#             results.append(trial[0])
#             pbar.update(1)
#     save_predict_result(results)
#
#
# def save_predict_result(results):
#     # 读取结果
#     test_df = pd.read_csv('resource/demo/test.csv')
#     # 填充结果
#     test_df['Prediction'] = results
#     # 　提取ID和预测结果两列
#     test_df = test_df[['QID', 'Prediction']]
#     # 保存结果.
#     test_df.to_csv('resource/demo/test_results.csv', index=None, sep=',')


if __name__ == '__main__':
    test('我的帕萨特烧机油怎么办')
