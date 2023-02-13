import torch
from torch.utils.data import DataLoader
from rspf_dpf import *
from utils import *
# import datetime

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = torch.device('cpu')

# start_time = datetime.datetime.now()

# dyn=Mark/Poly, prop=Boot/Uni/Deter, re=sys/mul
T = 50
N_p_train = 200
N_p_test = 2000
# dyn_list = ["Mark", "Poly"]
# prop_list = ["Boot", "Uni", "Deter"]
# re = "mul"
# gamma_list = [0.0, 0.5, 1.0]

P, A, B, C, D, beta = create_parameters(N_m=8)

# paracre_time = datetime.datetime.now()

# dataload_time = datetime.datetime.now()

trainingset = LoadTrainSet(dir="Poly_0.5")
train_data = DataLoader(dataset=trainingset, batch_size=100, shuffle=True)
validationset = LoadValSet(dir="Poly_0.5")
val_data = DataLoader(dataset=validationset, batch_size=500, shuffle=True)
rsdpf = RSDPF(rs=True, nf=True, tran_matrix=P, beta=beta).to(device)
rsdpf.train(train_data, val_data, N_iter=40, N_p=N_p_train, dyn="Poly", prop="Boot", re="mul")  
testset = LoadTestSet(dir="Poly_0.5")
test_data = DataLoader(dataset=testset, batch_size=500)
rsdpf.test(test_data, N_p=N_p_test, dyn="Poly", prop="Boot", re="mul")

# rs_list = [True, False]

# for rs in rs_list: 
#     for dyn in dyn_list: 
#         trainingset = LoadTrainSet(dir=dyn)
#         train_data = DataLoader(dataset=trainingset, batch_size=100, shuffle=True)
#         validationset = LoadValSet(dir=dyn)
#         val_data = DataLoader(dataset=validationset, batch_size=500, shuffle=True)
#         for prop in prop_list: 
#             rsdpf = RSDPF(rs=rs, nf=False, tran_matrix=P, beta=beta).to(device)
#             rsdpf.train(train_data, val_data, N_iter=40, N_p=N_p_train, dyn=dyn, prop=prop, re=re)
#             testset = LoadTestSet(dir=dyn)
#             test_data = DataLoader(dataset=testset, batch_size=500)
#             rsdpf.test(test_data, N_p=N_p_test, dyn=dyn, prop=prop, re=re)

# # training_time = datetime.datetime.now()

# rspf = RSPF(A, B, C, D, tran_matrix=P, beta=beta)
# for dyn in dyn_list: 
#     testset = LoadTestSet(dir=dyn)
#     test_data = DataLoader(dataset=testset, batch_size=500)
#     for prop in prop_list: 
#         rspf.test(test_data, N_p=N_p_test, dyn=dyn, prop=prop, re=re)

# for gamma in gamma_list: 
#     mmpf = MMPF(A, B, C, D, gamma=torch.tensor(gamma))
#     for dyn in dyn_list: 
#         testset = LoadTestSet(dir=dyn)
#         test_data = DataLoader(dataset=testset, batch_size=500)
#         mmpf.test(test_data, N_p=N_p_test, re=re, dyn=dyn)

# # test_time = datetime.datetime.now()

# # print(f"create parameter time:{(paracre_time - start_time).seconds}, load testdata time: {(dataload_time - paracre_time).seconds}, training time: {(training_time - dataload_time).seconds}, testing time: {(testing_time - traing_time).seconds}")






