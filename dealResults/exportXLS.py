import os
import xlwt

problems = {"possion", "burgers"}
optimizers = {"Adam_BFGS", "BFGS", "Adam"}


def result_dict(lines):
    """
    取出Nf_xxx_logs.txt文件内容写入字典res中
    :param lines: Nf_xxx_logs.txt文件所有行
    :return:      {"Nf":xxx, "L2 error":xxx, "training_time":xxx}
    """
    res = dict()
    for line in lines:
        splitLine = line.split("=")
        res[splitLine[0]] = splitLine[1].strip("\n")
    return res


def export_xls_file(problem, optimizer):
    """
    将所有结果导出为xls文件
    :param problem:    可选problems中的量
    :param optimizer:  可选optimizers中的量
    :return:           None
    """
    if problem not in problems:
        print("Can't export this problem!")
        return None
    if optimizer not in optimizers:
        print("Can't export this optimizer!")
        return None
    wb = xlwt.Workbook()
    sh1 = wb.add_sheet('Error')
    sh2 = wb.add_sheet('trainingTime')
    for j in range(len(Nfs)):
        sh1.write(0, j + 1, Nfs[i])
        sh2.write(0, j + 1, Nfs[i])

    # 将所有模型的所有Nf点的实验结果导出
    root = '..'
    target = 'Results/%s/%s' % (problem, optimizer)
    root_list = os.listdir(root)
    model_index = 0
    for file in root_list:
        # ../VgEmPINN
        PATH = root + '/' + file
        if os.path.isdir(PATH):
            model_list = os.listdir(PATH)
            if 'Results' in model_list:
                # 写入模型名字
                model_index += 1
                model_name = file
                sh1.write(model_index, 0, model_name)
                sh2.write(model_index, 0, model_name)
                # 找到包含所有结果的文件夹
                # '../VgEmPINN/Results/burgers/Adam_BFGS'
                PATH = PATH + '/' + target
                result_list = os.listdir(PATH)
                for result_folder in result_list:
                    # '../VgEmPINN/Results/burgers/Adam_BFGS/Nf5000_tanh_iter33113'
                    PATH_new = PATH + '/' + result_folder
                    results = os.listdir(PATH_new)
                    for resultFile in results:
                        # 找到.txt文件，取出结果
                        if os.path.splitext(resultFile)[1] == '.txt':
                            PATH_nn = PATH_new + '/' + resultFile
                            with open(PATH_nn) as f:
                                lines = f.readlines()
                                res_dict = res_dict(lines)
                                # 写入结果
                                sh1.write(model_index, Nf_col[res_dict['Nf']],
                                          float(res_dict['L2 error']))
                                sh2.write(model_index, Nf_col[res_dict['Nf']],
                                          float(res_dict['training_time']))
    wb.save('results_%s_%s.xls' % (problem, optimizer))
    print("Export done!")


if __name__ == '__main__':
    # problem = "burgers"
    prob = "possion"
    opt = "Adam_BFGS"
    # possion方程的Nf点设置
    if prob == "possion":
        Nfs = range(10, 101, 10)
        Nf_col = dict()
        ii = 1
        for i in range(10, 101, 10):
            Nf_col[str(i)] = ii
            ii += 1
    # burgers方程的Nf点设置
    if prob == "burgers":
        Nfs = [500, 1000, 2500, 5000, 10000]
        Nf_col = {"500": 1, "1000": 2, "2500": 3, "5000": 4, "10000": 5}
    export_xls_file(prob, opt)
