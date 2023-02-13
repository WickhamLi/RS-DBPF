import torch
import numpy as np
from torch.utils.data import Dataset

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = torch.device('cpu')

def create_parameters(N_m=8): 
    P = torch.zeros(N_m, N_m, device=device)
    for i in range(N_m): 
        if not i: 
            P += torch.diag(torch.tensor([0.80]*N_m, device=device), i)
        elif i==1: 
            P += torch.diag(torch.tensor([0.15]*(8-i), device=device), i)
            P += torch.diag(torch.tensor([0.15]*i, device=device), i-8)
        else: 
            P += torch.diag(torch.tensor([1/120]*(8-i), device=device), i)
            P += torch.diag(torch.tensor([1/120]*i, device=device), i-8)

    A = torch.tensor([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], device=device)
    B = torch.tensor([0., -2., 2., -4., 0., 2., -2., 4.], device=device)
    C = A.clone()
    D = B.clone()
    beta = torch.tensor([1.]*N_m, device=device)
    return P, A, B, C, D, beta

def data_generation(T, model, batch=1, dyn="Mark"): 
    m = torch.zeros(batch, 1, T, dtype=int)
    s = torch.zeros(batch, 1, T)
    o = torch.zeros(batch, 1, T)
    m[:, :, 0], s[:, :, 0] = model.initial(size=(batch, 1))
    o[:, :, 0] = model.obs(m[:, :, 0], s[:, :, 0])
    for t in range(1, T):
        if dyn=="Poly": 
            m[:, :, t] = model.Polyaurn_dynamic(m[:, :, :t])
        else: 
            m[:, :, t] = model.Markov_dynamic(m[:, :, t-1])
        s[:, :, t] = model.state(m[:, :, t], s[:, :, t-1])
        o[:, :, t] = model.obs(m[:, :, t], s[:, :, t])

    return m, s, o

class LoadTrainSet(Dataset): 
    def __init__(self, dir): 
        m_data = np.loadtxt(f'./datasets/{dir}/m_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.m = torch.from_numpy(m_data[:1000, np.newaxis, 1:])
        s_data = np.loadtxt(f'./datasets/{dir}/s_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.s = torch.from_numpy(s_data[:1000, np.newaxis, 1:])
        o_data = np.loadtxt(f'./datasets/{dir}/o_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.o = torch.from_numpy(o_data[:1000, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadValSet(Dataset): 
    def __init__(self, dir): 
        m_data = np.loadtxt(f'./datasets/{dir}/m_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.m = torch.from_numpy(m_data[1000:, np.newaxis, 1:])
        s_data = np.loadtxt(f'./datasets/{dir}/s_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.s = torch.from_numpy(s_data[1000:, np.newaxis, 1:])
        o_data = np.loadtxt(f'./datasets/{dir}/o_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.o = torch.from_numpy(o_data[1000:, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadTestSet(Dataset): 
    def __init__(self, dir): 
        m_data = np.loadtxt(f'./datasets/{dir}/m_test', delimiter=',', skiprows=1, dtype=np.float32)
        self.m = torch.from_numpy(m_data[:, np.newaxis, 1:])
        s_data = np.loadtxt(f'./datasets/{dir}/s_test', delimiter=',', skiprows=1, dtype=np.float32)
        self.s = torch.from_numpy(s_data[:, np.newaxis, 1:])
        o_data = np.loadtxt(f'./datasets/{dir}/o_test', delimiter=',', skiprows=1, dtype=np.float32)
        self.o = torch.from_numpy(o_data[:, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

def weights_bootstrap(model, w_list, m_list, s_list, o): 
    lw = torch.log(w_list)
    # if s_list.requires_grad: 
    #     s_list = torch.where(s_list==0., 1e-4, s_list)
    # s_obs = s_list.clone().detach()
    o = o.tile(1, s_list.shape[1], 1) - (model.co_C[m_list] * torch.sqrt(torch.abs(s_list)) + model.co_D[m_list])
    lw += torch.distributions.normal.Normal(loc=0., scale=model.sigma_v).log_prob(o)
#     for i in range(len(s_list)): 
#         lw[i] += stats.norm(loc=model.co_C[m_list[i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[i]], scale=np.sqrt(0.1)).logpdf(o)
# #         w += 10**(-300)
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True)


def weights_proposal(model, w_list, m_list, s_list, o, dyn): 
    lw = torch.log(w_list)
    o = o.tile(1, s_list.shape[1], 1) - (model.co_C[m_list[:, :, [-1]]] * torch.sqrt(torch.abs(s_list)) + model.co_D[m_list[:, :, [-1]]])
    if dyn=="Mark": 
        lp = torch.log(model.mat_P[m_list[:, :, [-2]], m_list[:, :, [-1]]])
    else: 
        batch, N_p = m_list.shape[0], m_list.shape[1]
        alpha = torch.zeros(batch, N_p, len(model.beta))
        for m_index in range(len(model.beta)): 
            alpha[:, :, m_index] = torch.sum(m_list[:, :, :-1] == m_index, dim=2)
        beta = model.beta[None, None, :]
        prob = (alpha + beta)/(alpha + beta).sum(dim=2, keepdim=True)
        batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
        pars_index = torch.arange(N_p, dtype=torch.long).tile(batch, 1).reshape(-1)
        index_flatten = m_list[:, :, [-1]].view(-1)
        lp = torch.log(prob[batch_index, pars_index, index_flatten].view(batch, N_p, 1))

    lw += torch.distributions.normal.Normal(loc=0., scale=0.1**(0.5)).log_prob(o) + lp - torch.log(torch.tensor(1/len(model.co_A)))
    # for i in range(len(s_list)): 
    #     lw[i] += stats.norm(loc=model.co_C[m_list[-1, i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[-1, i]], scale=np.sqrt(0.1)).logpdf(o) + lp[i] - np.log((1/len(model.co_A)))
#         w += 10**(-300)
#         print(lw[i])
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True)


def weights_mmpf(model, w_list, s_list, pi_list, o): 
    lw = torch.log(w_list)
    o = o.tile(1, s_list.shape[1], s_list.shape[2])[:, :, :, None] - (model.co_C * torch.sqrt(torch.abs(s_list)) + model.co_D)
    lw += torch.distributions.normal.Normal(loc=0., scale=model.sigma_v).log_prob(o)
    w = torch.exp(lw - lw.max(dim=2, keepdim=True)[0])
    pi_list *= w.sum(dim=2, keepdim=True) 
    return w/w.sum(dim=2, keepdim=True), pi_list/pi_list.sum(dim=1, keepdim=True)

def dyn_density(model, m, s_t, s_p, logdetJ=torch.Tensor([0])): 
    s_t -= (model.co_A[m] * s_p + model.co_C[m])
    log_den = torch.distributions.Normal(loc=0., scale=model.sigma_u).log_prob(s_t)
    if logdetJ.sum().item(): 
        log_den -= logdetJ
    return log_den

def obs_density(z, logdetJ): 
    log_z = torch.distributions.Normal(loc=0., scale=1.).log_prob(z)
    log_den = log_z + logdetJ

    return log_den

def weights_CNFs(w, logden_dyn, logden_prop, logden_obs): 
    lw = torch.log(w)
    # o = o.tile(1, s.shape[1], 1) - (model.co_C[m] * torch.sqrt(torch.abs(s)) + model.co_D[m])
    # logden_obs = torch.distributions.normal.Normal(loc=0., scale=model.sigma_v).log_prob(o)
    lw += (logden_dyn + logden_obs - logden_prop)
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True)

def inv_cdf(cum, ws): 
    index = torch.ones(ws.shape, dtype=torch.long, device=device) * cum.shape[-1]
    w_cum = ws.cumsum(dim=1).clone().detach()
    for i in range(cum.shape[-1]): 
        index[:, [i]] -= (cum[:, [i]] < w_cum).sum(dim=1, keepdim=True, dtype=torch.long)
        # while cum[i] > w: 
        #     k += 1
        #     w += ws[k]
        # index[i] = k
    index = torch.where(index < ws.shape[-1], index, ws.shape[-1]-1)
    return index

def inv_cdf_mmpf(cum, ws): 
    index = torch.ones(ws.shape, dtype=torch.long, device=device) * cum.shape[-1]
    w_cum = ws.cumsum(dim=2).clone().detach()
    for i in range(cum.shape[-1]): 
        index[:, :, [i]] -= (cum[:, :, [i]] < w_cum).sum(dim=2, keepdim=True, dtype=torch.long)
    index = torch.where(index < ws.shape[-1], index, ws.shape[-1]-1)
    return index.view(-1, ws.shape[-1])

def resample_systematic(ws): 
    batch, N_p = ws.shape
    cum = (torch.rand(batch, 1) + torch.arange(N_p).tile(batch, 1)) / torch.ones(batch, 1)*N_p   
    
    return inv_cdf(cum, ws)

def resample_systematic_mmpf(ws, batch): 
    N_p = ws.shape[-1]
    ws = ws.view(batch, -1, N_p)
    N_m = ws.shape[1]
    cum = (torch.rand(batch, N_m, 1) + torch.arange(N_p).tile(batch, N_m, 1)) / torch.ones(batch, N_m, 1)*N_p   
    return inv_cdf_mmpf(cum, ws)
    
def resample_multinomial(ws): 
    batch, N_p = ws.shape
    uni = (-torch.log(torch.rand(batch, N_p+1, device=device))).cumsum(dim=1)
    cum = uni[:, :-1] / uni[:, [-1]]

    return inv_cdf(cum, ws)

def resample_multinomial_mmpf(ws, batch): 
    N_p = ws.shape[-1]
    ws = ws.view(batch, -1, N_p)
    N_m = ws.shape[1]
    uni = (-torch.log(torch.rand(batch, N_m, N_p+1, device=device))).cumsum(dim=2)
    cum = uni[:, :, :-1] / uni[:, :, [-1]]

    return inv_cdf_mmpf(cum, ws)

