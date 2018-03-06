import math
import pdb
import traceback

import numpy as np
import tensorflow as tf
import progressbar


class MemN2N(object):
    def __init__(self, config, sess):
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
        self.pad_idx = config.pad_idx
        self.pre_trained_context_wt = config.pre_trained_context_wt
        # self.pre_trained_target_wt = config.pre_trained_target_wt

        self.input = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, 3], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")

        self.show = config.show

        self.hid = []

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std), trainable=False)
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std), trainable=False)
        self.ASP = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std), trainable=False)
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))
        self.BL_W = tf.Variable(tf.random_normal([2 * self.edim, 1], stddev=self.init_std))
        self.BL_B = tf.Variable(tf.zeros([1, 1]))

        # Location Encoding
        # self.T_A = tf.Variable(tf.random_normal([self.mem_size + 1, self.edim], stddev=self.init_std))
        # self.T_B = tf.Variable(tf.random_normal([self.mem_size + 1, self.edim], stddev=self.init_std))
        location_encoding = 1 - tf.truediv(self.time, self.mem_size)
        location_encoding = tf.cast(location_encoding, tf.float32)
        location_encoding3dim = tf.tile(tf.expand_dims(location_encoding, 2), [1, 1, self.edim])

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        # Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Ain = Ain_c * location_encoding3dim

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        # Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        Bin = Bin_c * location_encoding3dim

        ASPin = tf.nn.embedding_lookup(self.ASP, self.input)
        ASPinAgg = tf.reduce_mean(ASPin, axis=1)
        ASPout2dim = tf.reshape(ASPinAgg, [-1, self.edim])
        self.hid.append(ASPout2dim)

        for h in range(self.nhop):
            '''
            Bi-linear scoring function for a context word and aspect term
            '''
            til_hid = tf.tile(self.hid[-1], [1, self.mem_size])
            til_hid3dim = tf.reshape(til_hid, [-1, self.mem_size, self.edim])
            a_til_concat = tf.concat([til_hid3dim, Ain], axis=2)
            til_bl_wt = tf.tile(self.BL_W, [self.batch_size, 1])
            til_bl_3dim = tf.reshape(til_bl_wt, [self.batch_size, -1, 2 * self.edim])
            att = tf.matmul(a_til_concat, til_bl_3dim, adjoint_b=True)
            til_bl_b = tf.tile(self.BL_B, [self.batch_size, self.mem_size])
            til_bl_3dim = tf.reshape(til_bl_b, [-1, self.mem_size, 1])
            g = tf.nn.tanh(tf.add(att, til_bl_3dim))
            g_2dim = tf.reshape(g, [-1, self.mem_size])
            P = tf.nn.softmax(g_2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], self.C)
            Dout = tf.add(Cout, Bout2dim)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim - self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat([F, K], axis=1))

    def build_model(self):
        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, 3], stddev=self.init_std))
        z = tf.matmul(self.hid[-1], self.W)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=self.target)

        self.lr = tf.Variable(self.current_lr, trainable=False)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.W, self.ASP, self.BL_W, self.BL_B]
        grads_and_vars = self.opt.compute_gradients(self.loss, params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                  for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()

        self.training_summary = tf.summary.scalar("training_loss", tf.reduce_mean(self.loss))
        self.validation_summary = tf.summary.scalar("validation_loss", tf.reduce_mean(self.loss))

        self.correct_prediction = tf.argmax(z, 1)
        # op to write logs to Tensorboard
        logs_path = '/tmp/tensorflow_logs/example/'
        self.summary_writer = tf.summary.FileWriter(logs_path, self.sess.graph)

    def train(self, data):
        source_data, source_loc_data, target_data, target_label, _ = data
        N = int(math.floor(len(source_data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.mem_size], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, 3])  # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        if self.show:
            bar = progressbar.ProgressBar(max_value=N)

        np.random.seed(100)
        rand_idx, cur = np.random.permutation(len(source_data)), 0
        for idx in range(N):  # for each batch
            if self.show: bar.start()

            # initialize for each batch
            x.fill(self.pad_idx)
            context.fill(self.pad_idx)
            time.fill(self.mem_size)
            target.fill(0)

            '''
            Initilialize all the padding vector to 0 before backprop.
            TODO: Code is 5x slower due to the following initialization.
            '''

            for b in range(self.batch_size):
                if cur >= len(rand_idx): break

                m = rand_idx[cur]
                target[b][target_label[m]] = 1
                time[b, :len(source_loc_data[m])] = source_loc_data[m]
                x[b, :len(target_data[m])] = target_data[m]
                # x[b][0] = target_data[m]
                context[b, :len(source_data[m])] = source_data[m]
                cur = cur + 1

            def get_val(var):
                evaled = self.sess.run([
                    var],
                    feed_dict={
                        self.input: x,
                        self.time: time,
                        self.target: target,
                        self.context: context})
                return evaled

            try:
                loss, _, step, summary = self.sess.run([
                    self.loss,
                    self.optim,
                    self.global_step,
                    self.training_summary],
                    feed_dict={
                        self.input: x,
                        self.time: time,
                        self.target: target,
                        self.context: context})
            except Exception as e:
                traceback.print_exc()
                print(e)
                pdb.set_trace()

            # Write logs at every iteration
            self.summary_writer.add_summary(summary, step)
            cost += np.sum(loss)

        if self.show: bar.finish()
        _, train_acc = self.test(data)  # todo: bring me back
        return cost / N / self.batch_size, train_acc

    def test(self, data):
        source_data, source_loc_data, target_data, target_label, _ = data
        N = int(math.ceil(len(source_data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, 3])  # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])
        context.fill(self.pad_idx)

        m, acc = 0, 0
        for i in range(N):

            x.fill(self.pad_idx)
            context.fill(self.pad_idx)
            time.fill(self.mem_size)
            target.fill(0)

            raw_labels = []
            for b in range(self.batch_size):
                if m >= len(target_label): break

                target[b][target_label[m]] = 1
                # x[b][0] = target_data[m]
                x[b, :len(target_data[m])] = target_data[m]
                time[b, :len(source_loc_data[m])] = source_loc_data[m]
                context[b, :len(source_data[m])] = source_data[m]
                raw_labels.append(target_label[m])
                m += 1

            loss, summary, step = self.sess.run([self.loss, self.validation_summary, self.global_step],
                                                feed_dict={
                                                    self.input: x,
                                                    self.time: time,
                                                    self.target: target,
                                                    self.context: context})
            self.summary_writer.add_summary(summary, step)
            cost += np.sum(loss)

            predictions = self.sess.run(self.correct_prediction, feed_dict={self.input: x,
                                                                            self.time: time,
                                                                            self.target: target,
                                                                            self.context: context})

            for b in range(self.batch_size):
                if b >= len(raw_labels): break
                if raw_labels[b] == predictions[b]:
                    acc = acc + 1

        return cost / float(len(source_data)), acc / float(len(source_data))

    def run(self, train_data, test_data):
        print('training...')
        self.sess.run(self.A.assign(self.pre_trained_context_wt))
        self.sess.run(self.B.assign(self.pre_trained_context_wt))
        self.sess.run(self.ASP.assign(self.pre_trained_context_wt))

        saver = tf.train.Saver(tf.trainable_variables())
        for idx in range(self.nepoch):
            print('epoch ' + str(idx) + '...')
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            print('train-loss=%.2f;train-acc=%.2f;test_loss=%.2f;test-acc=%.2f;' % (
            train_loss, train_acc, test_loss, test_acc))
            # save_path = saver.save(self.sess, "./models/model.ckpt", global_step=idx)
            # print("Model saved in file: %s" % save_path)
