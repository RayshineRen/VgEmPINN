from PINN import PhysicsInformedNN
from Burgers import PINN_burgers
from Possion import PINN_possion
import tensorflow as tf
import numpy as np
from scipy.special import gamma, jacobi, roots_jacobi


def Jacobi(n, a, b, x):
    """
    Recursive generation of the Jacobi polynomial of order n
    :param n:
    :param a:
    :param b:
    :param x:
    :return:
    """
    x = np.array(x)
    return jacobi(n, a, b)(x)


def Test_fcn(N_test, x):
    """
    构造测试函数集
    :param N_test:
    :param x:
    :return:
    """
    test_total = []
    for n in range(1, N_test + 1):
        test = Jacobi(n + 1, 0, 0, x) - Jacobi(n - 1, 0, 0, x)
        test_total.append(test)
    return np.asarray(test_total)


def GaussLobattoJacobiWeights(Q: int, a, b):
    X = roots_jacobi(Q - 2, a + 1, b + 1)[0]
    if a == 0 and b == 0:
        W = 2 / ((Q - 1) * Q * (Jacobi(Q - 1, 0, 0, X) ** 2))
        Wl = 2 / ((Q - 1) * Q * (Jacobi(Q - 1, 0, 0, -1) ** 2))
        Wr = 2 / ((Q - 1) * Q * (Jacobi(Q - 1, 0, 0, 1) ** 2))
    else:
        W = 2 ** (a + b + 1) * gamma(a + Q) * gamma(b + Q) / (
                (Q - 1) * gamma(Q) * gamma(a + b + Q + 1) * (Jacobi(Q - 1, a, b, X) ** 2))
        Wl = (b + 1) * 2 ** (a + b + 1) * gamma(a + Q) * gamma(b + Q) / (
                (Q - 1) * gamma(Q) * gamma(a + b + Q + 1) * (Jacobi(Q - 1, a, b, -1) ** 2))
        Wr = (a + 1) * 2 ** (a + b + 1) * gamma(a + Q) * gamma(b + Q) / (
                (Q - 1) * gamma(Q) * gamma(a + b + Q + 1) * (Jacobi(Q - 1, a, b, 1) ** 2))
    W = np.append(W, Wr)
    W = np.append(Wl, W)
    X = np.append(X, 1)
    X = np.append(-1, X)
    return [X, W]


def generate_quad_data(Nx_quad, Ny_quad):
    [x_quad, w_quad_x] = GaussLobattoJacobiWeights(Nx_quad, 0, 0)
    [y_quad, w_quad_y] = GaussLobattoJacobiWeights(Ny_quad, 0, 0)
    # ++++++++++++++++++++++++++++
    # Quadrature points
    x_quad = x_quad[:, None]
    w_quad_x = w_quad_x[:, None]
    y_quad = y_quad[:, None]
    w_quad_y = w_quad_y[:, None]
    return x_quad, w_quad_x, y_quad, w_quad_y


class VPINN(PhysicsInformedNN):
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                 x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob):
        PhysicsInformedNN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended)
        self.xquad = x_quad  # x维度的求积点
        self.yquad = y_quad  # y维度的求积点
        self.wquad_x = w_x_quad  # x维度的权函数
        self.wquad_y = w_y_quad  # y维度的权函数
        self.f_ext_total = f_exact_total  # f(x,y)即等号右侧 二重积分后的值
        self.grid_x = grid_x  # x维度分界点集合，VPINN中就是x的值域
        self.grid_y = grid_y  # y维度分界点集合，VPINN中就是y的值域
        self.prob = prob  # problem
        self.NEx = grid_x.shape[0] - 1  # 将x维度分为NEx个区域
        self.NEy = grid_y.shape[0] - 1  # 将y维度分为NEy个区域
        self.loss_v_log = []
        self.loss_v = self.variational_loss()
        self.loss = self.loss_b + self.loss_r + self.loss_v

    def net_du(self, x, y):
        """
        计算二重积分用到的微分项
        :param x:
        :param y:
        :return:
        """
        u = self.net_u(x, y)
        d1ux = tf.gradients(u, x)[0]
        d1uy = tf.gradients(u, y)[0]
        d2ux = tf.gradients(d1ux, x)[0]
        d2uy = tf.gradients(d1uy, y)[0]
        return d1ux, d2ux, d1uy, d2uy

    def variational_loss(self):
        varloss_total = 0
        for e_y in range(self.NEy):
            for e_x in range(self.NEx):
                F_ext_element = self.f_ext_total[e_y*self.NEy + e_x]  # 定位子区域(VPINN只有一个区域)，此形式中f_ext恒0
                Ntest_element = int(np.sqrt(np.shape(F_ext_element)[0]))  # 子区域的测试函数个数 x=y
                x_quad_element = tf.constant(self.grid_x[e_x] + (self.grid_x[e_x + 1] - self.grid_x[e_x])
                                             / 2 * (self.xquad + 1))  # 将求积点映射到子区域区间内
                jacobian_x = (self.grid_x[e_x + 1] - self.grid_x[e_x]) / 2  # 系数
                # 测试函数及其微分 global(用xquad计算)
                testx_quad_element = Test_fcn(Ntest_element, self.xquad)

                y_quad_element = tf.constant(self.grid_y[e_y] + (self.grid_y[e_y + 1] - self.grid_y[e_y])
                                             / 2 * (self.yquad + 1))
                jacobian_y = (self.grid_y[e_y + 1] - self.grid_y[e_y]) / 2
                testy_quad_element = Test_fcn(Ntest_element, self.yquad)
                # PDE及其微分
                u_NN_quad_element = self.net_u(x_quad_element, y_quad_element)
                d1ux_NN_quad_element, d2ux_NN_quad_element, \
                    d1uy_NN_quad_element, d2uy_NN_quad_element = self.net_du(x_quad_element, y_quad_element)

                if self.prob == "burgers":
                    U_NN_element = []
                    # 计算二重积分
                    for phi_y in testy_quad_element:  # 对y积分
                        for phi_x in testx_quad_element:  # 对x积分
                            inte1_x = jacobian_x * tf.reduce_sum(
                                self.wquad_x * (d1uy_NN_quad_element + u_NN_quad_element * d1ux_NN_quad_element -
                                                0.01 / np.pi * d2ux_NN_quad_element) * phi_x)  # 权函数 * PDE * 测试函数
                            inte2_x = jacobian_y * tf.reduce_sum(self.wquad_y * inte1_x * phi_y)
                            U_NN_element.append(inte2_x)
                    U_NN_element = tf.reshape(U_NN_element, (-1, 1))
                elif self.prob == "possion":
                    U_NN_element_x = []
                    U_NN_element_y = []
                    for phi_y in testy_quad_element:
                        for phi_x in testx_quad_element:
                            inte1_x = jacobian_x * tf.reduce_sum(self.wquad_x * d2ux_NN_quad_element * phi_x)
                            inte1_y = jacobian_x * tf.reduce_sum(self.wquad_x * d2uy_NN_quad_element * phi_x)
                            inte2_x = jacobian_y * tf.reduce_sum(self.wquad_y * inte1_x * phi_y)
                            inte2_y = jacobian_y * tf.reduce_sum(self.wquad_y * inte1_y * phi_y)
                            U_NN_element_x.append(inte2_x)
                            U_NN_element_y.append(inte2_y)
                    U_NN_element_x = tf.reshape(U_NN_element_x, (-1, 1))
                    U_NN_element_y = tf.reshape(U_NN_element_y, (-1, 1))
                    U_NN_element = U_NN_element_x + U_NN_element_y
                Res_NN_element = U_NN_element - F_ext_element
                varloss = tf.reduce_mean(tf.square(Res_NN_element))
                varloss_total += varloss
        return varloss_total


class VPINN_burgers(PINN_burgers, VPINN):
    """
    从PINN_burgers继承net_f
    从VPINN继承__init__
    """

    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                 x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob):
        VPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                       x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob)


class VPINN_possion(PINN_possion, VPINN):
    """
    从PINN_possion继承net_f
    从VPINN继承__init__
    """

    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                 x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob):
        VPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                       x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob)
