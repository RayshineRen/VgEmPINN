import tensorflow as tf
import numpy as np
import time


def xavier_init(size):
    """
    xavier初始化
    :param size:
    :return:
    """
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float64)


def initialize_nn(layers):
    """
    神经网络权重和偏置的初始化，可以使用xavier_init
    :param layers:
    :return:
    """
    weights = []
    biases = []
    num_layers = len(layers)
    for layer in range(0, num_layers - 1):
        w = xavier_init(size=[layers[layer], layers[layer + 1]])
        b = tf.Variable(tf.zeros([1, layers[layer + 1]], dtype=tf.float64), dtype=tf.float64)
        weights.append(w)
        biases.append(b)
    return weights, biases


class PhysicsInformedNN:
    def __init__(self, x_u, u, x_f, layers, activation, lr, opt, extended):
        self.x_u = x_u[:, 0:1]
        self.t_u = x_u[:, 1:2]
        self.u = u
        self.x_f = x_f[:, 0:1]
        self.t_f = x_f[:, 1:2]
        self.layers = layers
        self.loss_log = []
        self.loss_r_log = []
        self.loss_b_log = []
        self.Nf = x_f.shape[0] - x_u.shape[0]
        self.activation = activation
        self.opt = opt
        self.extended = extended
        self.training_time = 0

        self.weights, self.biases = initialize_nn(layers)

        self.sess = tf.Session()

        self.x_u_tf = tf.placeholder(tf.float64, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float64, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float64, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float64, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss_b = tf.reduce_mean(tf.square(self.u_pred - self.u_tf))
        self.loss_r = tf.reduce_mean(tf.square(self.f_pred))
        self.loss = self.loss_b + self.loss_r

        self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                     method='L-BFGS-B',
                                                                     options={'maxiter': 50000,
                                                                              'maxfun': 50000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 1.0 * np.finfo(float).eps})
        self.LR = lr
        self.optimizer_Adam = tf.train.AdamOptimizer(self.LR)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def neural_net(self, x, weights, biases):
        num_layers = len(weights) + 1
        h = x
        for layer in range(0, num_layers - 2):
            w = weights[layer]
            b = biases[layer]
            h = self.activation(tf.add(tf.matmul(h, w), b))
        w = weights[-1]
        b = biases[-1]
        y = tf.add(tf.matmul(h, w), b)
        return y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u * u_x - 0.01 / np.pi * u_xx
        return f

    def callback(self, loss_value, loss_valueb, loss_valuer):
        self.loss_log.append(loss_value)
        self.loss_b_log.append(loss_valueb)
        self.loss_r_log.append(loss_valuer)
        iters = len(self.loss_log)
        if iters % 1000 == 0:
            str_print = 'It: %d, Lossb: %.3e, Lossr: %.3e'
            print(str_print % (iters, loss_valueb, loss_valuer))

    def optimize_bfgs(self, tf_dict, start_time):
        self.optimizer_BFGS.minimize(self.sess,
                                     feed_dict=tf_dict,
                                     fetches=[self.loss, self.loss_b, self.loss_r],
                                     loss_callback=self.callback)
        end_time = time.time()
        loss_value = self.sess.run(self.loss, tf_dict)
        print("training time %f, loss %f" % (end_time - start_time, loss_value))
        return end_time

    def optimize_adam(self, niter, tf_dict, start_time):
        for it in range(niter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            lb = self.sess.run(self.loss_b, tf_dict)
            lr = self.sess.run(self.loss_r, tf_dict)
            self.loss_log.append(loss_value)
            self.loss_b_log.append(lb)
            self.loss_r_log.append(lr)
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                str_print = 'It: %d, Lossb: %.3e, Lossr: %.3e, Time: %.2f'
                print(str_print % (it, lb, lr, elapsed))
        end_time = time.time()
        return end_time

    def train(self, niter=20000):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        start_time = time.time()  # 记录开始时间
        end_time = time.time()    # init结束时间
        print("\n Start training!\n optimizer is %s" % self.opt)
        if self.opt == 'BFGS':
            end_time = self.optimize_bfgs(tf_dict, start_time)
        elif self.opt == 'Adam':
            end_time = self.optimize_adam(niter, tf_dict, start_time)
        elif self.opt == 'Adam_BFGS':
            self.optimize_adam(niter // 2 + 1, tf_dict, start_time)
            print("Adam training done!\n Start BFGS training!")
            end_time = self.optimize_bfgs(tf_dict, start_time)
        self.training_time = end_time - start_time

    def predict(self, x_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: x_star[:, 0:1], self.t_u_tf: x_star[:, 1:2]})
        return u_star
