def logTime(model, problem, optimizer):
    if hasattr(model, 'training_time'):
        with open("./Results/RVB/%s/%s/Nf%d_logs.txt"%(problem, optimizer, model.Nf), 'a') as fout:
            fout.write("Nf=%d\n"%(model.Nf))
            fout.write("training_time=%f\n"%(model.training_time))
    print("training time appended done!")

def logRelativeError(model, error, problem, optimizer):
    with open("./Results/RVB/%s/%s/Nf%d_logs.txt"%(problem, optimizer, model.Nf), 'a') as fout:
        fout.write("L2 error=%f\n"%(error))
    print("L2 error appended done!")