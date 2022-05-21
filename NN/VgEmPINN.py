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
        EmPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, extended, prob)
        VPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, prob,
                       x_quad, w_x_quad, y_quad, w_y_quad, f_exact_total, grid_x, grid_y)
        gPINN.__init__(self, x_ibc, u, x_res, layers, maxIter, activation, lr, opt, prob,
                       w_x, w_t)
        self.loss = self.wb * self.loss_b + self.wr * self.loss_r + self.loss_v + self.loss_g
