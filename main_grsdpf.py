import torch
from rspf_torch import *
from utils import *
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

P, A, B, C, D, beta = create_parameters(N_m=8)
rspf = RSPF(P, A, B, C, D, beta=beta)
    
T = 50
N_p = 200
run = 2000

m, s, o = generate_data(T, rspf, batch=run, dyn="Poly")

m_np = m.numpy().reshape(run, T)
m_df = pd.DataFrame(m_np)
m_train, m_test = train_test_split(m_df, test_size=0.25, shuffle=False)
s_np = s.numpy().reshape(run, T)
s_df = pd.DataFrame(s_np)
s_train, s_test = train_test_split(s_df, test_size=0.25, shuffle=False)
o_np = o.numpy().reshape(run, T)
o_df = pd.DataFrame(o_np)
o_train, o_test = train_test_split(o_df, test_size=0.25, shuffle=False)

m_train.to_csv('./datasets/m_train')
s_train.to_csv('./datasets/s_train')
o_train.to_csv('./datasets/o_train')

m_test.to_csv('./datasets/m_test')
s_test.to_csv('./datasets/s_test')
o_test.to_csv('./datasets/o_test')


trainingset = LoadTrainSet()
train_data = DataLoader(dataset=trainingset, batch_size=100, shuffle=True)
validationset = LoadValSet()
val_data = DataLoader(dataset=validationset, batch_size=100, shuffle=True)


# # dyn=Mark/Poly, prop=Boot/Uni/Deter, re=sys/mul
rsdpf = RSDPF(P, beta=beta)
loss = rsdpf.training(train_data, val_data, N_p=N_p, dyn="Poly")
print(loss)

testset = LoadTestSet()
test_data = DataLoader(dataset=testset, batch_size=500)
mse_dpf, mse_pf = rsdpf.testing(test_data, A, B, C, D, dyn="Poly")
mse_dpf_df = pd.DataFrame(mse_dpf)
mse_pf_df = pd.DataFrame(mse_pf)
mse_dpf_df.to_csv('./results/mse_dpf')
mse_pf_df.to_csv('./results/mse_pf')


