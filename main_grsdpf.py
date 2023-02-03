# from rspf_np import *
from rspf_torch import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

P, A, B, C, D, beta = create_parameters()
rspf = RSPF(P, A, B, C, D, beta=beta)
    
T = 50
N_p = 2000
run = 1000

m_data, s_data, o_data = generate_data(T, rspf, batch=run, dyn="Mark")

o_data.to(device)
s_data.to(device)

s_train = s_data[:950, :]
s_test = s_data[-50:, :]
o_train = o_data[:950, :]
o_test = o_data[-50:, :]

# dyn=Mark/Poly, prop=Boot/Uni/Deter, re=sys/mul

rsdpf = RSDPF(P, beta=beta)
loss = rsdpf.training(s_train, o_train)
# m_parlist, s_parlist, w_parlist = filtering(rspf, o_data.reshape(run, -1), N_p=N_p, dyn="Mark", prop="Boot", re="mul")
# mse, mse_cum = MSE(s_parlist, s_data.reshape(run, T), w_parlist)
rsdpf.testing(s_test, o_test, loss)


