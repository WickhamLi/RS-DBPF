# from rspf_np import *
from rspf_torch import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

P, A, B, C, D, beta = create_parameters()
rspf = RSPF(P, A, B, C, D, beta=beta).to(device)
    
T = 50
N_p = 2000
run = 500

# m_data = np.zeros((run, T), dtype=int)
# s_data = np.zeros((run, T))
# o_data = np.zeros((run, T))

m_data, s_data, o_data = generate_data(T, rspf, batch=run, dyn="Mark")

# np.random.seed(222)
# for i in range(run): 
#     m, s, o = generate_data(T, rspf, size=run)
#     m_SMC[i, :] = np.array(m).T
#     s_SMC[i, :] = np.array(s).T
#     o_SMC[i, :] = np.array(o).T

# m_parlist = np.zeros((run, N_p, T))
# s_parlist = np.zeros((run, N_p, T))
# w_parlist = np.zeros((run, N_p, T))
# mse_list = np.zeros((run, T))
# msecum_list = np.zeros((run, T))

# dyn=Mark/Poly, prop=Boot/Uni/Deter, re=sys/mul
m_parlist, s_parlist, w_parlist = filtering(rspf, o_data.reshape(run, -1), N_p=N_p, dyn="Mark", prop="Boot", re="mul")
mse, mse_cum = MSE(s_parlist, s_data.reshape(run, T), w_parlist)

# for i in range(run): 
#     m_list, s_list, w_list = filtering(T, rspf, m_SMC[i, 0], s_SMC[i, 0], o_SMC[i, :], N_p=N_p, dyn=0)
#     mlist_SMC[i, :] = m_list
#     slist_SMC[i, :] = s_list
#     wlist_SMC[i, :] = w_list
#     mse_SMC[i, :], msecum_SMC[i, :] = MSE(wlist_SMC[i, :], slist_SMC[i, :], s_SMC[i, :])

plt.plot(mse_cum.mean(dim=0), label='RSPF(Bootstrap)')
plt.ylabel('Average Cumulative MSE')
plt.xlabel('Time Step')
plt.yscale('log')
plt.legend()
plt.tight_layout()
# plt.savefig('RSPFMark.png')
plt.show()

print(mse_cum.mean(dim=0)[-1])
print(mse.mean(), mse.max(), mse.min())
