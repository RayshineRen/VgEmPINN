from Burgers import PINN_burgers
from Possion import PINN_possion
from PINN import PhysicsInformedNN
import tensorflow as tf


class gPINN(PhysicsInformedNN):
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended, w_x, w_t):
        PhysicsInformedNN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended)
        self.loss_g_log = []
        self.w_x = w_x
        self.w_t = w_t
        self.loss_g = self.gradient_enhance()
        self.loss = self.loss_b + self.loss_r + self.loss_g

    def gradient_enhance(self):
        """
        gradient-enhanced
        :return:
        """
        g_x = tf.reduce_mean(tf.square(tf.gradients(self.f_pred, self.x_res_tf)[0]))
        g_t = tf.reduce_mean(tf.square(tf.gradients(self.f_pred, self.t_res_tf)[0]))
        return self.w_x * g_x + self.w_t * g_t


class gPINN_burgers(PINN_burgers, gPINN):
    """
    从PINN_burgers继承net_f
    从gPINN继承gradient-enhanced
    """
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended, w_x, w_t):
        gPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended, w_x, w_t)


class gPINN_possion(PINN_possion, gPINN):
    """
    从PINN_possion继承net_f
    从gPINN继承gradient-enhanced
    """
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended, w_x, w_t):
        gPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended, w_x, w_t)
