import sys
sys.path.insert(0, '../../Storage/')
sys.path.insert(0, '../../NN/')
sys.path.insert(0, '../../trainingSet/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
import numpy as np
from scipy.interpolate import griddata
from storage_utils import dumpTotalLoss
from log_utils import logTime, logRelativeError
from plot_utils import plotting
from file_utils import arrangeFiles
from PINN import PhysicsInformedNN
from Possion import PossionData, u_ext

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
    extended = "square"
    # possion Data
    data = PossionData(x_l, x_r, y_l, y_h)
    data.generate_ibc(n_u)
    data.generate_res(n_f)
    # PINN for possion equation
    model = PhysicsInformedNN(data.x_u_train, data.u_train, data.x_f, layers,
                              maxIter, activation, lr, opt, extended)
    model.train()
    # Test point
    x = np.linspace(x_l, x_r, n_test)
    y = np.linspace(y_l, y_h, n_test)
    X, Y = np.meshgrid(x, y)
    u = u_ext(X, Y)
    data = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
    u_pred = model.predict(np.hstack((X.flatten()[:, None], Y.flatten()[:, None])))
    U_pred = griddata(data, u_pred.flatten(), (X, Y), method='cubic')
    error_u = np.linalg.norm(u_pred - u.flatten()[:, None]) / np.linalg.norm(u.flatten()[:, None])
    # 将数据记录到磁盘中
    if not os.path.exists("./Results/%s/%s" % (problem, opt)):
        os.makedirs("./Results/%s/%s" % (problem, opt))
    logTime(model, problem, opt)
    logRelativeError(model, error_u, problem, opt)
    plotting(X, Y, u.T, U_pred.T, problem, opt)
    dumpTotalLoss(model, problem, opt)
    niter = len(model.loss_log)
    arrangeFiles(model, niter, problem, opt)
