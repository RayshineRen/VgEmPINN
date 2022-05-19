import os
import time
import shutil

def mkdir(Nf, act, iter, PATH):
    '''
    创建以Nfxx_tanh_iterxxx命名的文件夹
    :param Nf:
    :param act:
    :param iter:
    :param PATH:    父文件夹
    :return:
    '''
    new_path = PATH + "/Nf%d_%s_iter%d"%(Nf, act, iter)
    folder = os.path.exists(new_path)
    if not folder:
        os.makedirs(new_path)
        print("mkdir done!")
    else:
        print("folder exists!")
        print("make folder by time")
        new_path = new_path + str(time.time()).split(".")[0]
        os.makedirs(new_path)
        print("mkdir done!")
    return new_path

def movefiles(old, new):
    '''
    将生成在父目录的文件全部移动到mkdir函数创建的新文件夹中
    :param old:
    :param new:
    :return:
    '''
    filelist = os.listdir(old)
    for file in filelist:
        src = old + "/" + file
        if os.path.isfile(src):
            dst = new + "/" + file
            shutil.move(src, dst)
    print("files moved done!")

def arrangeFiles(model, iter, problem, optimizer):
    '''
    整理文件夹
    :param model:
    :param iter:
    :param problem:
    :param optimizer:
    :return:
    '''
    PATH = "./Results/%s/%s"%(problem, optimizer)
    new = mkdir(model.Nf, model.activation._tf_api_names[-1], iter, PATH)
    movefiles(PATH, new)
    print("File arranged done!")
