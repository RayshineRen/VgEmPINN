# mark NN Storage trainingSet as source root
# import sys
# sys.path.insert(0, '../../Storage/')
# sys.path.insert(0, '../../NN/')
# sys.path.insert(0, '../../trainingSet/')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu
import tensorflow as tf
from Burgers import BurgersData
from PINN import PhysicsInformedNN

if __name__ == '__main__':
    # hyper parameters
    problem = "burgers"
    n_u = 100
    n_f = 500
    layers = [2] + [40] * 5 + [1]
    maxIter = 20000
    activation = tf.tanh
    lr = 0.001
    opt = "Adam_BFGS"
    # burgers Data
    data = BurgersData(n_u)
    data.generate_res_data(n_f)
    # PINN for burgers' equation
    model = PhysicsInformedNN(data.x_u_train, data.u_train, data.x_f, layers,
                              maxIter, activation, lr, opt, problem)
    data.run_model(model)
