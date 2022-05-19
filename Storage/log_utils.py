def logTime(model, problem, optimizer):
    '''
    将trainingTime记录到log文件中
    :param model:       训练好的模型
    :param problem:     e.g. burgers
    :param optimizer:   e.g. optimizer
    :return:            None
    '''
    if hasattr(model, 'training_time'):
        with open("./Results/%s/%s/Nf%d_logs.txt"%(problem, optimizer, model.Nf), 'a') as fout:
            fout.write("Nf=%d\n"%(model.Nf))
            fout.write("training_time=%f\n"%(model.training_time))
    print("training time appended done!")

def logRelativeError(model, error, problem, optimizer):
    '''
    将L2 error记录到log文件中
    :param model:       训练好的模型
    :param error:       L2 error
    :param problem:     e.g. burgers
    :param optimizer:   e.g. optimizer
    :return:            None
    '''
    with open("./Results/%s/%s/Nf%d_logs.txt"%(problem, optimizer, model.Nf), 'a') as fout:
        fout.write("L2 error=%f\n"%(error))
    print("L2 error appended done!")