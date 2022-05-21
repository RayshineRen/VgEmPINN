import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
from Possion import PossionData
from VPINN import VPINN, generate_quad_data
import numpy as np

if __name__ == '__main__':
    # hyper parameter
    problem = "possion"
    n_u = 20
    n_f = 50
    n_test = 100
    layers = [2] + [20] * 3 + [1]
    maxIter = 1000
    activation = tf.tanh
    [x_l, x_r] = [-1, 1]
    [y_l, y_h] = [-1, 1]
    lr = 0.001
    opt = 'Adam_BFGS'
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
    # PINN for possion equation
    model = VPINN(data.x_u_train, data.u_train, data.x_f, layers,
                  maxIter, activation, lr, opt, problem,
                  x_quad, w_quad_x, y_quad, w_quad_y, f_ext_total, grid_x, grid_y)
    data.run_model(model)
