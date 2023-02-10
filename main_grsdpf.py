import torch
from torch.utils.data import DataLoader
from generate_data import create_parameters
from rspf_dpf import *
from utils import *
import datetime

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = torch.device('cpu')

start_time = datetime.datetime.now()

T = 50
N_p_train = 200
N_p_test = 2000
dyn = "Poly"
prop = "Boot"
re = "mul"

P, A, B, C, D, beta = create_parameters(N_m=8)

paracre_time = datetime.datetime.now()

testset = LoadTestSet(dir=dyn)
test_data = DataLoader(dataset=testset, batch_size=500)

loadtest_time = datetime.datetime.now()

rspf = RSPF(A, B, C, D, tran_matrix=P, beta=beta)
rspf.testing(test_data, N_p=N_p_test, dyn=dyn, prop=prop, re=re)

rspf_time = datetime.datetime.now()

print(f"create parameter time:{(paracre_time - start_time).seconds}, load testdata time: {(loadtest_time - paracre_time).seconds}, rspf opeartion time: {(rspf_time - loadtest_time).seconds}")

# trainingset = LoadTrainSet()
# train_data = DataLoader(dataset=trainingset, batch_size=100, shuffle=True)
# validationset = LoadValSet()
# val_data = DataLoader(dataset=validationset, batch_size=500, shuffle=True)

# # dyn=Mark/Poly, prop=Boot/Uni/Deter, re=sys/mul
# rsdpf = RSDPF(P, beta, rs=False, nf=False).to(device)
# rsdpf.training(rsdpf, train_data, val_data, N_p=N_p, dyn="Poly", nf=False)
# print(loss)


# mse_dpf, mse_pf = testing(rsdpf, test_data, A, B, C, D, dyn="Poly", nf=False)
# mse_dpf_df = pd.DataFrame(mse_dpf)
# mse_dpf_df.to_csv('./results/mse_dpf')





