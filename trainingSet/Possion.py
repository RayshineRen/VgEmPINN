import numpy as np
from pyDOE import lhs
from scipy.interpolate import griddata
import os
from storage_utils import dumpTotalLoss
from log_utils import logTime, logRelativeError
from plot_utils import plotting
from file_utils import arrangeFiles
from PINN import PhysicsInformedNN
import tensorflow as tf


def u_ext(x, y):
    """
    steep solution u(x)
    :param x:
    :param y:
    :return:
    """
    u = 2*(1+y)/((3+x)**2+(1+y)**2)
    return u


class PossionData:
    def __init__(self, x_l, x_r, y_l, y_h):
        self.x_l = x_l
        self.x_r = x_r
        self.y_l = y_l
        self.y_h = y_h
        self.problem = "possion"

    def generate_ibc(self, n_u):
        """
        生成四条边界
        :return:
        """
        # x = 1
        X_u_train_x1 = np.array(n_u * [1])[:, None]
        X_u_train_y1 = np.linspace(self.y_l, self.y_h, n_u)[:, None]
        X_u_train_1 = np.hstack((X_u_train_x1, X_u_train_y1))
        u_train_1 = u_ext(X_u_train_x1, X_u_train_y1)
        # x = -1
        X_u_train_x2 = np.array(n_u * [-1])[:, None]
        X_u_train_y2 = np.linspace(self.y_l, self.y_h, n_u)[:, None]
        X_u_train_2 = np.hstack((X_u_train_x2, X_u_train_y2))
        u_train_2 = u_ext(X_u_train_x2, X_u_train_y2)
        # y = 1
        X_u_train_x3 = np.linspace(self.x_l, self.x_r, n_u)[:, None]
        X_u_train_y3 = np.array(n_u * [1])[:, None]
        X_u_train_3 = np.hstack((X_u_train_x3, X_u_train_y3))
        u_train_3 = u_ext(X_u_train_x3, X_u_train_y3)
        # y = -1
        X_u_train_x4 = np.linspace(self.x_l, self.x_r, n_u)[:, None]
        X_u_train_y4 = np.array(n_u * [-1])[:, None]
        X_u_train_4 = np.hstack((X_u_train_x4, X_u_train_y4))
        u_train_4 = u_ext(X_u_train_x4, X_u_train_y4)
        # 合并
        X_u_train = np.vstack((X_u_train_1, X_u_train_2,
                               X_u_train_3, X_u_train_4))
        u_train = np.vstack((u_train_1, u_train_2,
                             u_train_3, u_train_4))
        self.x_u_train = X_u_train
        self.u_train = u_train

    def generate_res(self, n_f):
        """
        生成collocation points
        :return:
        """
        lb = np.array([self.x_l, self.y_l])
        ub = np.array([self.x_r, self.y_h])
        self.x_f = lb + (ub - lb) * lhs(2, n_f)

    def run_model(self, model, n_test=100):
        """
        在possion数据集上训练模型
        :param model:
        :param n_test:
        :return:
        """
        model.train()
        # Test point
        x = np.linspace(self.x_l, self.x_r, n_test)
        y = np.linspace(self.y_l, self.y_h, n_test)
        X, Y = np.meshgrid(x, y)
        u = u_ext(X, Y)
        data = np.hstack([X.flatten()[:, None], Y.flatten()[:, None]])
        u_pred = model.predict(np.hstack((X.flatten()[:, None], Y.flatten()[:, None])))
        U_pred = griddata(data, u_pred.flatten(), (X, Y), method='cubic')
        error_u = np.linalg.norm(u_pred - u.flatten()[:, None]) / np.linalg.norm(u.flatten()[:, None])
        # 将数据记录到磁盘中
        if not os.path.exists("./Results/%s/%s" % (self.problem, model.opt)):
            os.makedirs("./Results/%s/%s" % (self.problem, model.opt))
        logTime(model, self.problem, model.opt)
        logRelativeError(model, error_u, self.problem, model.opt)
        plotting(X, Y, u.T, U_pred.T, self.problem, model.opt)
        dumpTotalLoss(model, self.problem, model.opt)
        niter = len(model.loss_log)
        arrangeFiles(model, niter, self.problem, model.opt)
