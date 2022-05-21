from Burgers import PINN_burgers
from Possion import PINN_possion
from gPINN import gPINN
from VPINN import VPINN
from EmPINN import EmPINN


class VgEmPINN(EmPINN, VPINN, gPINN):
    """
    三合一究极缝合模型
    """
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                 x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob,
                 w_x, w_t):
        EmPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended)
        VPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                       x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob)
        gPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                       w_x, w_t)
        self.loss = self.wb * self.loss_b + self.wr * self.loss_r + self.loss_v + self.loss_g


class VgEmPINN_burgers(VgEmPINN, PINN_burgers):
    """
    从PINN_burgers继承net_f
    从VgEmPINN继承新的属性等
    """
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                 x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob,
                 w_x, w_t):
        VgEmPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                          x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob,
                          w_x, w_t)


class VgEmPINN_possion(VgEmPINN, PINN_possion):
    """
    从PINN_possion继承net_f
    从VgEmPINN继承新的属性等
    """
    def __init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                 x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob,
                 w_x, w_t):
        VgEmPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended,
                          x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y, prob,
                          w_x, w_t)
