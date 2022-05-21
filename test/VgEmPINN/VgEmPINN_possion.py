import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
import numpy as np
from Possion import PossionData
from VgEmPINN import VgEmPINN_possion
from VPINN import generate_quad_data


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
    # gPINN
    w_x = 0.001
    w_t = 0.001
    # VPINN
    Nx_testfcn = 5
    Ny_testfcn = 5
    Nx_Quad = 50
    Ny_Quad = 50
    x_quad, w_quad_x, y_quad, w_quad_y = generate_quad_data(Nx_Quad, Ny_Quad)
    grid_x = np.array([x_l, x_r])
    grid_y = np.array([y_l, y_h])
    f_ext = np.array((Nx_testfcn * Ny_testfcn) * [0])[:, None]
    f_ext_total = np.tile(f_ext, (1, 1)).reshape(-1, Nx_testfcn * Ny_testfcn, 1)
    # possion Data
    data = PossionData(x_l, x_r, y_l, y_h)
    data.generate_ibc(n_u)
    data.generate_res(n_f)
    # VgEmPINN for possion equation
    model = VgEmPINN_possion(data.x_u_train, data.u_train, data.x_f, layers,
                             maxIter, activation, lr, opt, extended,
                             x_quad, w_quad_x, y_quad, w_quad_y, f_ext_total, grid_x, grid_y, data.problem,
                             w_x, w_t)
    data.run_model(model)
