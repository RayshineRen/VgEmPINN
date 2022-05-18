import os
import xlwt

def resultDict(lines):
    res = dict()
    for line in lines:
        splitLine = line.split("=")
        res[splitLine[0]] = splitLine[1].strip("\n")
    return res

# problem = "burgers"
problem = "burgers"
optimizer = "Adam_BFGS"
if problem == "possion":
    Nfs = range(10, 101, 10)
    Nf_col = dict()
    ii = 1
    for i in range(10, 101, 10):
        Nf_col[str(i)] = ii
        ii += 1

if problem == "burgers":
    Nfs = [500, 1000, 2500, 5000, 10000]
    Nf_col = {"500": 1, "1000": 2, "2500": 3, "5000": 4, "10000": 5}

wb = xlwt.Workbook()
sh1 = wb.add_sheet('Error')
sh2 = wb.add_sheet('trainingTime')
for i in range(len(Nfs)):
    sh1.write(0, i+1, Nfs[i])
    sh2.write(0, i+1, Nfs[i])

root = '../'
target = 'Results/RVB/%s/%s'%(problem, optimizer)
root_list = os.listdir(root)
model_index = 0
for file in root_list:
    PATH = root + '/' + file
    if os.path.isdir(PATH):
        model_list = os.listdir(PATH)
        if 'Results' in model_list:
            # 写入模型名字
            model_index += 1
            model_name = file
            sh1.write(model_index, 0, model_name)
            sh2.write(model_index, 0, model_name)
            PATH = PATH + '/' + target
            result_list = os.listdir(PATH)
            for result_folder in result_list:
                PATH_new = PATH + '/' + result_folder
                results = os.listdir(PATH_new)
                for resultFile in results:
                    if os.path.splitext(resultFile)[1] == '.txt':
                        PATH_nn = PATH_new + '/' + resultFile
                        with open(PATH_nn) as f:
                            lines = f.readlines()
                            result_dict = resultDict(lines)
                            # 写入结果
                            sh1.write(model_index, Nf_col[result_dict['Nf']],\
                                      float(result_dict['L2 error']))
                            sh2.write(model_index, Nf_col[result_dict['Nf']], \
                                      float(result_dict['training_time']))

wb.save('results_%s_%s.xls'%(problem, optimizer))