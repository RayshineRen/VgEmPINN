import sys
sys.path.insert(0, '../../Storage/')
sys.path.insert(0, '../../NN/')
sys.path.insert(0, '../../trainingSet/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
import numpy as np
from storage_utils import dumpTotalLoss
from log_utils import logTime, logRelativeError
from plot_utils import plotting
from file_utils import arrangeFiles
from PINN import PhysicsInformedNN
from Burgers import BurgersData


if __name__ == '__main__':
    # hyper parameters
    prob = "burgers"
    n_u = 100
    n_f = 500
    layers = [2] + [40] * 5 + [1]
    maxIter = 20000
    activation = tf.tanh
    lr = 0.001
    opt = "Adam_BFGS"
    extended = "square"
    # burgers Data
    data = BurgersData(n_u)
    data.generate_res_data(n_f)
    # PINN for burgers' equation
    model = PhysicsInformedNN(data.x_u_train, data.u_train, data.x_f, layers,
                              maxIter, activation, lr, opt, extended)
    model.train()
    u_pred = model.predict(data.x_star)
    # 将数据记录到磁盘中
    if not os.path.exists("./Results/%s/%s" % (prob, opt)):
        os.makedirs("./Results/%s/%s" % (prob, opt))
    error_u = np.linalg.norm(data.u_star - u_pred, 2) / np.linalg.norm(data.u_star, 2)
    logTime(model, prob, opt)
    logRelativeError(model, error_u, prob, opt)
    u_pred = u_pred.reshape(-1, 256)
    plotting(data.X, data.T, data.Exact, u_pred, prob, opt)
    dumpTotalLoss(model, prob, opt)
    niter = len(model.loss_log)
    arrangeFiles(model, niter, prob, opt)

