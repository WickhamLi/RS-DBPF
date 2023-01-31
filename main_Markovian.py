from rspf import *

P, A, B, C, D, beta = create_parameters()
rspf = RSPF(P, A, B, C, D, beta=beta)
    
T = 50
N_p = 2000

run = 1
m_SMC = np.zeros((run, T))
s_SMC = np.zeros((run, T))
o_SMC = np.zeros((run, T))

# np.random.seed(222)
for i in range(run): 
    m, s, o = generate_data(T, rspf)
    m_SMC[i, :] = np.array(m).T
    s_SMC[i, :] = np.array(s).T
    o_SMC[i, :] = np.array(o).T

mlist_SMC = np.zeros((run, T, N_p))
slist_SMC = np.zeros((run, T, N_p))
wlist_SMC = np.zeros((run, T, N_p))
mse_SMC = np.zeros((run, T))
msecum_SMC = np.zeros((run, T))

for i in range(run): 
    m_list, s_list, w_list = filtering(T, rspf, m_SMC[i, 0], s_SMC[i, 0], o_SMC[i, :], N_p=N_p, dyn=3)
    mlist_SMC[i, :] = m_list
    slist_SMC[i, :] = s_list
    wlist_SMC[i, :] = w_list
    mse_SMC[i, :], msecum_SMC[i, :] = MSE(wlist_SMC[i, :], slist_SMC[i, :], s_SMC[i, :])

plt.plot(np.mean(msecum_SMC, axis=0), label='RSPF(Bootstrap)')
plt.ylabel('Average Cumulative MSE')
plt.xlabel('Time Step')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig('RSPFMarkov.png')
plt.show()

print(np.mean(msecum_SMC, axis=0)[-1])
