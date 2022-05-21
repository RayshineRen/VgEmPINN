from PINN import PhysicsInformedNN
import tensorflow as tf


class gPINN(PhysicsInformedNN):
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, prob, w_x, w_t):
        PhysicsInformedNN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, prob)
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
