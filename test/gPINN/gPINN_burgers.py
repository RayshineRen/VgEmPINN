import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
from gPINN import gPINN_burgers
from Burgers import BurgersData


if __name__ == '__main__':
    # hyper parameters
    n_u = 100
    n_f = 500
    layers = [2] + [40] * 5 + [1]
    maxIter = 20000
    activation = tf.tanh
    lr = 0.001
    opt = "Adam_BFGS"
    extended = "square"
    w_x = 0.001
    w_t = 0.001
    # burgers Data
    data = BurgersData(n_u)
    data.generate_res_data(n_f)
    # PINN for burgers' equation
    model = gPINN_burgers(data.x_u_train, data.u_train, data.x_f, layers,
                          maxIter, activation, lr, opt, extended, w_x, w_t)
    data.run_model(model)
