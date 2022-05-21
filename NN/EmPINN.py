from Burgers import PINN_burgers
from Possion import PINN_possion
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
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended):
        self.layers = layers
        self.weights, self.biases, \
            self.W1, self.b1, self.W2, self.b2 = initialize_mNN(self.layers)
        # PhysicsInformedNN.__init__会调用self.net_u,需要保证self.W1等属性已经在类中定义
        super(EmPINN, self).__init__(x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended)

    def net_u(self, x, y):
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


class EmPINN_burgers(PINN_burgers, EmPINN):
    """
    从PINN_burgers继承net_f
    从EmPINN继承新的属性, net_u, modified_neural_net
    """

    # 注意报错AttributeError: 'EmPINN_burgers' object has no attribute 'W1' 调用逻辑 EmPINN __init__函数
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended):
        EmPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended)


class EmPINN_possion(PINN_burgers, EmPINN):
    """
    从PINN_burgers继承net_f
    从EmPINN继承新的属性, net_u, modified_neural_net
    """

    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended):
        EmPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended)
