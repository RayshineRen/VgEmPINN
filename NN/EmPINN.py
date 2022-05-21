from PINN import PhysicsInformedNN, xavier_init
import tensorflow as tf


def initialize_mNN(layers):
    """
    modified NN 参数初始化
    :param layers:
    :return:
    """
    weights = []
    biases = []
    num_layers = len(layers)
    for layer in range(0, num_layers - 1):
        W = xavier_init(size=[layers[layer], layers[layer + 1]])
        b = tf.Variable(tf.zeros([1, layers[layer + 1]], dtype=tf.float64), dtype=tf.float64)
        weights.append(W)
        biases.append(b)
    # parameters of U
    W1 = xavier_init([layers[0], layers[1]])
    b1 = tf.Variable(tf.zeros([1, layers[1]], dtype=tf.float64), dtype=tf.float64)
    # parameters of V
    W2 = xavier_init([layers[0], layers[1]])
    b2 = tf.Variable(tf.zeros([1, layers[1]], dtype=tf.float64), dtype=tf.float64)
    return weights, biases, W1, b1, W2, b2


def modified_neural_net(X, weights, biases, activation, W1, b1, W2, b2):
    """
    modified PINN实现
    :param X:
    :param weights:
    :param biases:
    :param activation:
    :param W1:
    :param b1:
    :param W2:
    :param b2:
    :return:
    """
    num_layers = len(weights) + 1
    U = activation(tf.add(tf.matmul(X, W1), b1))
    V = activation(tf.add(tf.matmul(X, W2), b2))
    H = X
    for layer in range(0, num_layers - 2):
        W = weights[layer]
        b = biases[layer]
        H = activation(tf.add(tf.matmul(H, W), b))
        H = tf.multiply(1 - H, U) + tf.multiply(H, V)
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


class EmPINN(PhysicsInformedNN):
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended, prob):
        self.layers = layers
        self.weights, self.biases, \
            self.W1, self.b1, self.W2, self.b2 = initialize_mNN(self.layers)
        self.extended = extended  # 扩维方式
        # PhysicsInformedNN.__init__会调用self.net_u,需要保证self.W1等属性已经在类中定义
        PhysicsInformedNN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, prob)
        self.wb = 200  # IBC 权重
        self.wr = 10  # res 权重
        self.loss = self.wb * self.loss_b + self.wr * self.loss_r

    def net_u(self, x, y):
        """
        Extended PINN
        :param x:
        :param y:
        :return:
        """
        if self.extended == "square":
            u = modified_neural_net(tf.concat([x, y, x ** 2, y ** 2], 1), self.weights, self.biases,
                                    self.activation, self.W1, self.b1, self.W2, self.b2)
        elif self.extended == "inverse":
            u = modified_neural_net(tf.concat([x, y, 1 / x, 1 / y], 1), self.weights, self.biases,
                                    self.activation, self.W1, self.b1, self.W2, self.b2)
        elif self.extended == "quartic":
            u = modified_neural_net(tf.concat([x, y, x ** 4, y ** 4], 1), self.weights, self.biases,
                                    self.activation, self.W1, self.b1, self.W2, self.b2)
        elif self.extended == "sin":
            u = modified_neural_net(tf.concat([x, y, tf.sin(x), tf.sin(y)], 1), self.weights, self.biases,
                                    self.activation, self.W1, self.b1, self.W2, self.b2)
        elif self.extended == "cos":
            u = modified_neural_net(tf.concat([x, y, tf.cos(x), tf.cos(y)], 1), self.weights, self.biases,
                                    self.activation, self.W1, self.b1, self.W2, self.b2)
        else:
            self.layers[0] = 2
            u = modified_neural_net(tf.concat([x, y], 1), self.weights, self.biases,
                                    self.activation, self.W1, self.b1, self.W2, self.b2)
        return u
