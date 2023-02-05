from rspf_torch import *
from rspf_np import *
from torch.utils.data import Dataset, DataLoader

class LoadTestSet(Dataset): 
    def __init__(self): 
        m_data = np.loadtxt('./m_test', delimiter=',', dtype=np.int, skiprows=1)
        self.m = torch.from_numpy(m_data[:, np.newaxis, 1:])
        s_data = np.loadtxt('./s_test', delimiter=',', dtype=np.float, skiprows=1)
        self.s = torch.from_numpy(s_data[:, np.newaxis, 1:])
        o_data = np.loadtxt('./o_test', delimiter=',', dtype=np.float, skiprows=1)
        self.o = torch.from_numpy(o_data[:, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadTestResult(Dataset): 
    def __init__(self): 
        s_data = np.loadtxt('./testresult', delimiter=',', dtype=np.float, skiprows=1)
        self.s = torch.from_numpy(s_data[:, np.newaxis, 1:])
        self.nums = self.s.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

testset = LoadTestSet()
test_data = DataLoader(dataset=testset, batch_size=50)

P, A, B, C, D, beta = create_parameters()
rspf = RSPF(P, A, B, C, D, beta=beta)
for m_data, s_data, o_data in test_data: 
    m_parlist, s_parlist, w_parlist = filtering(rspf, o_data.reshape(50, -1).numpy(), N_p=200, dyn="Mark", prop="Boot", re="mul")
mse_pf, mse_cum_pf = MSE(s_parlist, s_data.reshape(50, 50).numpy(), w_parlist)

testresult = LoadTestResult()
test_result = DataLoader(dataset=testresult, batch_size=50)

for m_test, s_test, o_test in test_data: 
    rsdpf = RSDPF(torch.Tensor(P), A=torch.Tensor([0.8782, -0.2472,  0.2494, -0.1092, -0.0904,  0.0601,  0.0921, -0.7813]), B=torch.Tensor([ 3.8717,  2.5013, -3.1995,  1.2672, -2.8956, -0.3078,  2.9331, -3.4809]), C=torch.Tensor([ 0.7456, -1.0411,  1.1509, -0.1037,  0.7360,  0.6059,  0.1337, -0.3315]), D=torch.Tensor([ 3.8208,  2.4796, -3.2802,  1.1364, -3.0545, -0.5164,  2.9471, -3.5563]), sigma_u=torch.Tensor([0.4341]), sigma_v=torch.Tensor([0.2533]))
    s_est, _ = rsdpf.forward(m_test[:, :, [0]], s_test[:, :, [0]], o_test, N_p=200, dyn="Mark", prop="Boot", re="mul")

loss = ((s_est - s_test.to(device))**2).mean()
print(f'test loss = {loss:.8f}')
    
mse = ((s_est - s_test)**2).detach().numpy()
mse_cum = mse.cumsum(axis=-1)
plt.plot(mse_cum.mean(axis=0)[-1], label='RSDPF(Bootstrap)')
plt.plot(mse_cum_pf.mean(axis=0), label='RSPF(Bootstrap)')
plt.ylabel('Average Cumulative MSE')
plt.xlabel('Time Step')
plt.yscale('log')

plt.legend()
plt.tight_layout()
plt.savefig('RSDPFMarkComparison.png')
plt.show()
