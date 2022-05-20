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
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim), dtype=np.float64)
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=np.float64), dtype=tf.float64)


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
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended):
        # ibc points
        self.x_ibc = x_ibc[:, 0:1]
        self.t_ibc = x_ibc[:, 1:2]
        self.u = u
        # res points
        self.x_res = x_res[:, 0:1]
        self.t_res = x_res[:, 1:2]
        self.layers = layers
        self.maxIter = maxIter
        self.loss_log = []
        self.loss_r_log = []
        self.loss_b_log = []
        self.Nf = x_res.shape[0]  # 用于写入log.txt文件
        self.activation = activation
        self.opt = opt                             # 优化器选取
        self.extended = extended                   # 扩维方式
        self.training_time = 0

        self.weights, self.biases = initialize_nn(layers)

        self.sess = tf.Session()

        self.x_ibc_tf = tf.placeholder(tf.float64, shape=[None, self.x_ibc.shape[1]])
        self.t_ibc_tf = tf.placeholder(tf.float64, shape=[None, self.t_ibc.shape[1]])
        self.u_tf = tf.placeholder(tf.float64, shape=[None, self.u.shape[1]])
        self.x_res_tf = tf.placeholder(tf.float64, shape=[None, self.x_res.shape[1]])
        self.t_res_tf = tf.placeholder(tf.float64, shape=[None, self.t_res.shape[1]])

        self.u_pred = self.net_u(self.x_ibc_tf, self.t_ibc_tf)
        self.f_pred = self.net_f(self.x_res_tf, self.t_res_tf)

        self.loss_b = tf.reduce_mean(tf.square(self.u_pred - self.u_tf))
        self.loss_r = tf.reduce_mean(tf.square(self.f_pred))
        self.loss = self.loss_b + self.loss_r

        self.optimizer_BFGS = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                     method='L-BFGS-B',
                                                                     options={'maxiter': self.maxIter,
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
        """
        神经网络结构 全连接层
        :param x:
        :param weights:
        :param biases:
        :return:
        """
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

    def net_u(self, x, y):
        """
        u_NN
        :param x:
        :param t:
        :return:
        """
        u = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        """
        由PDE决定 重载
        :param x:
        :param t:
        :return:
        """
        pass

    def callback(self, *loss):
        """
        BFGS优化器callback函数
        :param loss:
        :return:
        """
        loss_value, loss_valueb, loss_valuer, *unpacked = loss  # 先解包所有模型都有的loss部分
        self.loss_log.append(loss_value)
        self.loss_b_log.append(loss_valueb)
        self.loss_r_log.append(loss_valuer)
        iters = len(self.loss_log)
        str_print = 'It: %d, Lossb: %.3e, Lossr: %.3e'
        # gPINN
        if hasattr(self, 'loss_g') and not hasattr(self, 'loss_v'):
            loss_valueg = unpacked[0]
            self.loss_g_log.append(loss_valueg)
            if iters % 500 == 0:
                str_print += ", Lossg: %.3e"
                print(str_print % (iters, loss_valueb, loss_valuer, loss_valueg))
        # VPINN
        elif hasattr(self, 'loss_v') and not hasattr(self, 'loss_g'):
            loss_valuev = unpacked[0]
            self.loss_v_log.append(loss_valuev)
            if iters % 500 == 0:
                str_print += ", Lossv: %.3e"
                print(str_print % (iters, loss_valueb, loss_valuer, loss_valuev))
        # VgPINN
        elif hasattr(self, 'loss_g') and hasattr(self, 'loss_v'):
            loss_valueg, loss_valuev = unpacked
            self.loss_g_log.append(loss_valueg)
            self.loss_v_log.append(loss_valuev)
            if iters % 500 == 0:
                str_print += ", Lossg: %.3e"
                str_print += ", Lossv: %.3e"
                print(str_print % (iters, loss_valueb, loss_valuer, loss_valueg, loss_valuev))
        # PINN
        else:
            if iters % 500 == 0:
                print(str_print % (iters, loss_valueb, loss_valuer))

    def optimize_bfgs(self, tf_dict, start_time):
        """
        BFGS优化
        :param tf_dict:
        :param start_time:
        :return:
        """
        fetches = [self.loss, self.loss_b, self.loss_r]
        # gPINN
        if hasattr(self, 'loss_g') and not hasattr(self, 'loss_v'):
            fetches.append(self.loss_g)
        # VPINN
        elif hasattr(self, 'loss_v') and not hasattr(self, 'loss_g'):
            fetches.append(self.loss_v)
        # VgPINN
        elif hasattr(self, 'loss_g') and hasattr(self, 'loss_v'):
            fetches.append(self.loss_g)
            fetches.append(self.loss_v)
        self.optimizer_BFGS.minimize(self.sess,
                                     feed_dict=tf_dict,
                                     fetches=fetches,
                                     loss_callback=self.callback)
        end_time = time.time()
        loss_value = self.sess.run(self.loss, tf_dict)
        print("training time %f, loss %f" % (end_time - start_time, loss_value))
        return end_time

    def optimize_adam(self, niter, tf_dict, start_time):
        """
        Adam优化
        :param niter:
        :param tf_dict:
        :param start_time:
        :return:
        """
        for it in range(niter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # 先获取所有模型共有的loss
            loss_value = self.sess.run(self.loss, tf_dict)
            lb = self.sess.run(self.loss_b, tf_dict)
            lr = self.sess.run(self.loss_r, tf_dict)
            self.loss_log.append(loss_value)
            self.loss_b_log.append(lb)
            self.loss_r_log.append(lr)
            elapsed = time.time() - start_time
            # gPINN
            if hasattr(self, 'loss_g') and not hasattr(self, 'loss_v'):
                lg = self.sess.run(self.loss_g, tf_dict)
                self.loss_g_log.append(lg)
                if it % 500 == 0:
                    str_print = 'It: %d, Lossb: %.3e, Lossr: %.3e, Lossg: %.3e, Time: %.2f'
                    print(str_print % (it, lb, lr, lg, elapsed))
            # VPINN
            elif hasattr(self, 'loss_v') and not hasattr(self, 'loss_g'):
                lv = self.sess.run(self.loss_v, tf_dict)
                self.loss_v_log.append(lv)
                if it % 500 == 0:
                    str_print = 'It: %d, Lossb: %.3e, Lossr: %.3e, Lossv: %.3e, Time: %.2f'
                    print(str_print % (it, lb, lr, lv, elapsed))
            # VgPINN
            elif hasattr(self, 'loss_g') and hasattr(self, 'loss_v'):
                lg = self.sess.run(self.loss_g, tf_dict)
                lv = self.sess.run(self.loss_v, tf_dict)
                self.loss_g_log.append(lg)
                self.loss_v_log.append(lv)
                if it % 500 == 0:
                    str_print = 'It: %d, Lossb: %.3e, Lossr: %.3e, Lossg: %.3e, Lossv: %.3e, Time: %.2f'
                    print(str_print % (it, lb, lr, lg, lv, elapsed))
            # PINN
            else:
                if it % 500 == 0:
                    str_print = 'It: %d, Lossb: %.3e, Lossr: %.3e, Time: %.2f'
                    print(str_print % (it, lb, lr, elapsed))
        end_time = time.time()
        return end_time

    def train(self):
        """
        训练神经网络
        :return:
        """
        niter = self.maxIter
        tf_dict = {self.x_ibc_tf: self.x_ibc, self.t_ibc_tf: self.t_ibc, self.u_tf: self.u,
                   self.x_res_tf: self.x_res, self.t_res_tf: self.t_res}
        start_time = time.time()  # 记录开始时间
        end_time = time.time()  # init结束时间
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
        """
        预测解
        :param x_star:
        :return:
        """
        u_star = self.sess.run(self.u_pred, {self.x_ibc_tf: x_star[:, 0:1], self.t_ibc_tf: x_star[:, 1:2]})
        return u_star
