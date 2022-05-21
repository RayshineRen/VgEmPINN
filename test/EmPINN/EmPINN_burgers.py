import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
from Burgers import BurgersData
from EmPINN import EmPINN_burgers

if __name__ == '__main__':
    # hyper parameters
    n_u = 100
    n_f = 500
    # Extended modified PINN
    layers = [4] + [40] * 5 + [1]
    maxIter = 20000
    activation = tf.tanh
    lr = 0.001
    opt = "Adam_BFGS"
    extended = "square"
    # burgers Data
    data = BurgersData(n_u)
    data.generate_res_data(n_f)
    # EmPINN for burgers' equation
    model = EmPINN_burgers(data.x_u_train, data.u_train, data.x_f, layers,
                           maxIter, activation, lr, opt, extended)
    data.run_model(model)
