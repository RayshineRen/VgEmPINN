import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
from Possion import PossionData
from EmPINN import EmPINN

if __name__ == '__main__':
    # hyper parameter
    problem = "possion"
    n_u = 20
    n_f = 50
    n_test = 100
    # EmPINN
    layers = [4] + [20] * 3 + [1]
    maxIter = 1000
    activation = tf.tanh
    [x_l, x_r] = [-1, 1]
    [y_l, y_h] = [-1, 1]
    lr = 0.001
    opt = 'Adam_BFGS'
    extended = "square"
    # possion Data
    data = PossionData(x_l, x_r, y_l, y_h)
    data.generate_ibc(n_u)
    data.generate_res(n_f)
    # EmPINN for possion equation
    model = EmPINN(data.x_u_train, data.u_train, data.x_f, layers,
                   maxIter, activation, lr, opt, extended, problem)
    data.run_model(model)
