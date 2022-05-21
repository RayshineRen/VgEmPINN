import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
import numpy as np
from VPINN import VPINN, generate_quad_data
from Burgers import BurgersData

if __name__ == '__main__':
    # hyper parameters
    problem = "burgers"
    n_u = 100
    n_f = 500
    layers = [2] + [40] * 5 + [1]
    maxIter = 20000
    activation = tf.tanh
    lr = 0.001
    opt = "Adam_BFGS"
    # VPINN
    Nx_testfcn = 5
    Ny_testfcn = 5
    Nx_Quad = 50
    Ny_Quad = 50
    x_quad, w_quad_x, y_quad, w_quad_y = generate_quad_data(Nx_Quad, Ny_Quad)
    grid_x = np.array([-1, 1])
    grid_y = np.array([0, 1])
    f_ext = np.array((Nx_testfcn * Ny_testfcn) * [0])[:, None]
    f_ext_total = np.tile(f_ext, (1, 1)).reshape(-1, Nx_testfcn * Ny_testfcn, 1)
    # burgers Data
    data = BurgersData(n_u)
    data.generate_res_data(n_f)
    # PINN for burgers' equation
    model = VPINN(data.x_u_train, data.u_train, data.x_f, layers,
                  maxIter, activation, lr, opt, problem,
                  x_quad, w_quad_x, y_quad, w_quad_y, f_ext_total, grid_x, grid_y)
    data.run_model(model)
