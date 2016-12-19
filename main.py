# The End-to-End Memory Network model is based on the paper "End-To-End Memory Networks" by Sainbayar
# Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus (https://arxiv.org/abs/1503.08895v4)
# --
# The implementation partially reuses classes and methods from End-To-End Memory Networks in Tensorflow
# (https://github.com/liangkai/MemN2N-tensorflow)
# =====================================================================================================

"""Lauch training ops."""
import os
import pprint
import tensorflow as tf
import pickle

from input_data import read_data, convert_question
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 6, "number of hops [6]")
flags.DEFINE_integer("mem_size", 100, "memory size [100]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", True, "print progress [False]")
flags.DEFINE_string("infere", "", "Question to answer, if any ['']")
flags.DEFINE_boolean("preloaded_data", False, "Whether to load presaved pickle [False]")

FLAGS = flags.FLAGS

def main(_):
    count = [] # List of (word, count) for all the data
    word2idx = {} # Dict (word, ID) for all the data

    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)

    # Lists of word IDs
    if FLAGS.preloaded_data:
        with open('preloaded_telenor/train.pickle', 'rb') as f:
            train_data = pickle.load(f)
        with open('preloaded_telenor/val.pickle', 'rb') as f:
            valid_data = pickle.load(f)
            word2idx = pickle.load(f)
    else:
        train_data = read_data('%s/train.pickle' % FLAGS.data_dir, count, word2idx)
        valid_data = read_data('%s/val.pickle' % FLAGS.data_dir, count, word2idx)
        if FLAGS.is_test:
            test_data = read_data('%s/test.pickle' % FLAGS.data_dir, count, word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.nwords = len(word2idx)

    pp.pprint(flags.FLAGS.__flags)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        # Build the Memory Network
        model = MemN2N(FLAGS, sess)
        model.build_model()

        if len(FLAGS.infere) > 0:
            print('Make sure the training and validation data supplied are the same as during the training of the model (idx2word)')
            question = convert_question(FLAGS.infere, word2idx)
            model.infere(question, idx2word) # Prediction
        elif FLAGS.is_test:
            model.run(valid_data, test_data, idx2word) # Testing
        else:
            model.run(train_data, valid_data, idx2word) # Training


if __name__ == '__main__':
    tf.app.run()
