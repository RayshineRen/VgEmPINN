import scipy.io
import numpy as np
from pyDOE import lhs


class BurgersData:
    def __init__(self, n_u):
        self.n_u = n_u
        self.load_data()

    def load_data(self):
        """
        导入Burgers方程的数值解
        :return:
        """
        path = r"D:\Documents\grade4term2\Graduate\VgEmPINN\Data"
        data = scipy.io.loadmat(path + "/burgers_shock.mat")
        x = data['x'].flatten()[:, None]
        t = data['t'].flatten()[:, None]
        Exact = np.real(data['usol']).T
        self.Exact = Exact
        X, T = np.meshgrid(x, t)
        self.X = X
        self.T = T
        self.x_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.u_star = Exact.flatten()[:, None]
        # Doman bounds
        self.lb = self.x_star.min(0)
        self.ub = self.x_star.max(0)
        # 初始点
        xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
        uu1 = Exact[0:1, :].T
        # x=-1的边界点
        xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
        uu2 = Exact[:, 0:1]
        # x=1的边界点
        xx3 = np.hstack((X[:, -1:], T[:, -1:]))
        uu3 = Exact[:, -1:]
        X_u_train = np.vstack([xx1, xx2, xx3])
        u_train = np.vstack([uu1, uu2, uu3])
        idx = np.random.choice(X_u_train.shape[0], self.n_u, replace=False)
        self.x_u_train = X_u_train[idx]
        self.u_train = u_train[idx]
        print("Burgers data loaded!")

    def generate_res_data(self, n_f):
        """
        生成collocation points
        :param n_f:
        :return:
        """
        x_f = self.lb + (self.ub - self.lb) * lhs(2, n_f)
        x_f = np.vstack((x_f, self.x_u_train))
        self.x_f = x_f
