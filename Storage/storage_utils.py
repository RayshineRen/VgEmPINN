import numpy as np

def dumpTotalLoss(model, problem, optimizer):
    '''
    将训练好的模型中的所有loss导出为npy文件
    :param model:
    :param problem:
    :param optimizer:
    :return:
    '''
    PATH = './Results/%s/%s'%(problem, optimizer)
    if hasattr(model, 'loss_log'):
        np.save(PATH + '/Nf%d_loss.npy' %(model.Nf), model.loss_log)
    if hasattr(model, 'loss_b_log'):
        np.save(PATH + '/Nf%d_loss_b.npy' %(model.Nf), model.loss_b_log)
    if hasattr(model, 'loss_r_log'):
        np.save(PATH + '/Nf%d_loss_r.npy' %(model.Nf), model.loss_r_log)
    if hasattr(model, 'loss_g_log'):
        np.save(PATH + '/Nf%d_loss_g.npy' %(model.Nf), model.loss_g_log)
    if hasattr(model, 'loss_v_log'):
        np.save(PATH + '/Nf%d_loss_v.npy' %(model.Nf), model.loss_v_log)
    print("dump to .npy done!")
