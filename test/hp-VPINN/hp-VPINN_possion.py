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
    NEx = 2
    NEy = 2
    Nx_testfcn = 5
    Ny_testfcn = 5
    Nx_Quad = 50
    Ny_Quad = 50
    x_quad, w_quad_x, y_quad, w_quad_y = generate_quad_data(Nx_Quad, Ny_Quad)
    # 区域划分
    delta_x = (x_r - x_l) / NEx
    # 区域网格点 eg. 均分为两个区域[-1, 0, 1]
    grid_x = np.asarray([x_l + i * delta_x for i in range(NEx + 1)])
    # 每个区域内测试函数的数目，目前都一样，结果只取决于区域内的求积点
    delta_y = (y_h - y_l) / NEy
    grid_y = np.asarray([y_l + i * delta_y for i in range(NEy + 1)])
    # 计算Fk NEx * NEy 个区域
    F_ext = np.array((Nx_testfcn * Ny_testfcn) * [0])[:, None]
    F_ext_total = np.tile(F_ext, (NEx * NEy, 1)).reshape(-1, Nx_testfcn * Ny_testfcn, 1)
    # possion Data
    data = PossionData(x_l, x_r, y_l, y_h)
    data.generate_ibc(n_u)
    data.generate_res(n_f)
    # PINN for possion equation
    model = VPINN(data.x_u_train, data.u_train, data.x_f, layers,
                  maxIter, activation, lr, opt, problem,
                  x_quad, w_quad_x, y_quad, w_quad_y, F_ext_total, grid_x, grid_y)
    data.run_model(model)
