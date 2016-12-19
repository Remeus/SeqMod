import os
import math
import numpy as np
import tensorflow as tf
from past.builtins import xrange
from scipy.ndimage.interpolation import shift
import json


LENGTH_PRED = 10 # Number of words that are being generated

class MemN2N(object):
    """Basic MemN2N"""

    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

        self.input = tf.placeholder(tf.float32, [None, self.edim], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")

        self.hid = []
        self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        """Build embedding components and other memory variables"""
        self.global_step = tf.Variable(0, name="global_step")

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))

        # Temporal Encoding
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        self.T_B = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        Bin = tf.add(Bin_c, Bin_t)

        for h in xrange(self.nhop):
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.batch_matmul(self.hid3dim, Ain, adj_y=True)
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.batch_matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], self.C)
            Dout = tf.add(Cout, Bout2dim)

            self.share_list[0].append(Cout)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat(1, [F, K]))

    def build_model(self):
        """Build and train the model through SGD"""
        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std))
        self.z = tf.matmul(self.hid[-1], self.W, name="output")

        self.loss = tf.nn.softmax_cross_entropy_with_logits(self.z, self.target)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()


    def train(self, data, idx2word):
        """Training ops."""
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        for idx in xrange(N-1): # Each batch ### Last examples discarded
            if self.show: bar.next()
            length_prediction = LENGTH_PRED
            cost_partial = 0
            indices_batch = list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
            last_context = np.zeros([self.batch_size, self.mem_size], dtype=np.int32)
            for i in range(self.batch_size):
                status = data[indices_batch[i]][0]
                for j in range(len(status)):
                    if j >= len(status) - self.mem_size:
                        last_context[i, j - len(status)] = status[j]
            for k in range(length_prediction): # Each word position
                target.fill(0)
                for b in xrange(self.batch_size): # Each couple status-answer
                    # Selection status used for training
                    index_status = indices_batch[b]
                    # Get status & 1st answer (list word IDs)
                    status = data[index_status][0]
                    answer = data[index_status][1]
                    # Prepare network
                    try:
                        target[b][answer[k]] = 1 # Prediction k-th word
                    except:
                        pass
                    context[b] = last_context[b, :]

                output, _, loss, self.step = self.sess.run([self.z,
                                                    self.optim,
                                                    self.loss,
                                                    self.global_step],
                                                    feed_dict={
                                                        self.input: x,
                                                        self.time: time,
                                                        self.target: target,
                                                        self.context: context})
                cost += np.sum(loss)
                cost_partial += np.sum(loss)

                ### Update context for batch
                for i in range(self.batch_size):
                    shift(last_context[i], -1, cval=0)
                    last_context[i, -1] = np.argmax(output, axis=1)[i]

            print(' | Batch %d / %d | Loss = %.2f' % (idx + 1, N - 1, cost_partial / LENGTH_PRED / self.batch_size))

        ### Print last example
        question_indices = status
        output_indices = last_context[-1, -length_prediction:]
        answer_indices = answer
        question_text = output_text = answer_text = " "
        for ind in question_indices:
            question_text += idx2word[ind] + " "
        for ind in output_indices:
            output_text += idx2word[ind] + " "
        for ind in answer_indices:
            answer_text += idx2word[ind] + " "
        print('\nQ: %s' % question_text)
        print('O: %s' % output_text)
        print('A: %s\n' % answer_text)

        if self.show: bar.finish()
        return cost / N / LENGTH_PRED / self.batch_size

    def test(self, data, idx2word):
        f = open('output.txt', 'a', encoding='utf_8')
        f.write('\n\n-- NEW EPOCH --\n\n')
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords])  # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:, t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        for idx in xrange(N - 1):  # Each batch ### Last examples discarded
            if self.show: bar.next()
            length_prediction = LENGTH_PRED
            cost_partial = 0
            indices_batch = list(range(idx * self.batch_size, (idx + 1) * self.batch_size))
            ### Status + last predicted words, for each row in the batch
            last_context = np.zeros([self.batch_size, self.mem_size], dtype=np.int32)
            for i in range(self.batch_size):
                status = data[indices_batch[i]][0]
                for j in range(len(status)):
                    if j >= len(status) - self.mem_size:
                        last_context[i, j - len(status)] = status[j]
            for k in range(length_prediction):  # Each word position
                target.fill(0)
                for b in xrange(self.batch_size):  # Each couple status-answer
                    ### Selection status used for training
                    index_status = indices_batch[b]
                    ### Get status & 1st answer (list word IDs)
                    status = data[index_status][0]
                    answer = data[index_status][1]
                    ### Prepare network
                    try:
                        target[b][answer[k]] = 1  # Prediction k-th word
                    except:
                        pass
                    context[b] = last_context[b, :]

                output, loss = self.sess.run([self.z,
                                              self.loss],
                                             feed_dict={
                                               self.input: x,
                                               self.time: time,
                                               self.target: target,
                                               self.context: context})
                cost += np.sum(loss)
                cost_partial += np.sum(loss)

                ### Update context for batch
                for i in range(self.batch_size):
                    shift(last_context[i], -1, cval=0)
                    last_context[i, -1] = np.argmax(output, axis=1)[i]

            print(' | Batch %d / %d | Loss = %.2f' % (idx + 1, N - 1, cost_partial / LENGTH_PRED / self.batch_size))
            output_indices = last_context[-1, -length_prediction:]
            output_text = ""
            for ind in output_indices:
                output_text += idx2word[ind] + " "
            f.write(output_text + '\n')


        ### Print last example
        question_indices = status
        answer_indices = answer
        question_text = answer_text = ""
        for ind in question_indices:
            question_text += idx2word[ind] + " "
        for ind in answer_indices:
            answer_text += idx2word[ind] + " "
        print('\nQ: %s' % question_text)
        print('O: %s' % output_text)
        print('A: %s\n' % answer_text)

        if self.show: bar.finish()
        return cost / N / LENGTH_PRED / self.batch_size

    def run(self, train_data, test_data, idx2word):
        if not self.is_test:
            for idx in xrange(self.nepoch):
                print('## Epoch %d / %d ##' % (idx+1, self.nepoch))
                train_loss = np.sum(self.train(train_data, idx2word))
                test_loss = np.sum(self.test(test_data, idx2word))

                # Logging
                self.log_loss.append([train_loss, test_loss])
                self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'perplexity': math.exp(train_loss),
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'valid_perplexity': math.exp(test_loss)
                }
                print(state)
                with open('stats.txt', 'a') as f:
                    f.write(json.dump(state) + '\n')

                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 10 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step = self.step.astype(int))
        else:
            self.load()

            valid_loss = np.sum(self.test(train_data, idx2word))
            test_loss = np.sum(self.test(test_data, idx2word))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Test mode but no checkpoint found")

    def infere(self, question, idx2word):
        self.load()
        answer = ""

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords])  # one-hot-encoded
        context = np.zeros([self.batch_size, self.mem_size], dtype=np.int32)

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:, t].fill(t)

        length_prediction = LENGTH_PRED

        for j in range(len(question)):
            if j >= len(question) - self.mem_size:
                context[0, j - len(question)] = question[j]

        for k in range(length_prediction):  # Each word position
            target.fill(0)
            output = self.sess.run(self.z,
                                 feed_dict={
                                     self.input: x,
                                     self.time: time,
                                     self.target: target,
                                     self.context: context})

            ### Update context for batch
            shift(context[0], -1, cval=0)
            context[0, -1] = np.argmax(output, axis=1)[0]

            answer += idx2word[context[0, -1]] + " "

        print(answer)
