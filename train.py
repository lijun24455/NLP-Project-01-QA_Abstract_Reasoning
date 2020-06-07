from seq2seq.model import Seq2Seq
from train_helper import *
from utils.config import *
from utils.vocab import Vocab


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count
    # params["trained_epoch"] = get_train_msg()
    params["learning_rate"] *= np.power(0.9, params["trained_epoch"])

    # 构建模型
    print("Building the model ...")
    model = Seq2Seq(params)
    # 获取保存管理者
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, SEQ2SEQ_CKPT, max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # 训练模型
    print("开始训练模型..")
    print("trained_epoch:", params["trained_epoch"])
    print("mode:", params["mode"])
    print("epochs:", params["epochs"])
    print("batch_size:", params["batch_size"])
    print("max_enc_len:", params["max_enc_len"])
    print("max_dec_len:", params["max_dec_len"])
    print("learning_rate:", params["learning_rate"])

    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':
    params = get_params()
    train(params)
