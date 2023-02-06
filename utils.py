import torch
import numpy as np
from torch.utils.data import Dataset

# device = torch.device('mps') if torch.backends.mps.is_available else torch.device('cpu')
device = torch.device('cpu')

class LoadTrainSet(Dataset): 
    def __init__(self): 
        m_data = np.loadtxt('./datasets/m_train', delimiter=',', skiprows=1)
        self.m = torch.from_numpy(m_data[:1400, np.newaxis, 1:])
        s_data = np.loadtxt('./datasets/s_train', delimiter=',', skiprows=1)
        self.s = torch.from_numpy(s_data[:1400, np.newaxis, 1:])
        o_data = np.loadtxt('./datasets/o_train', delimiter=',', skiprows=1)
        self.o = torch.from_numpy(o_data[:1400, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadValSet(Dataset): 
    def __init__(self): 
        m_data = np.loadtxt('./datasets/m_train', delimiter=',', skiprows=1)
        self.m = torch.from_numpy(m_data[1400:, np.newaxis, 1:])
        s_data = np.loadtxt('./datasets/s_train', delimiter=',', skiprows=1)
        self.s = torch.from_numpy(s_data[1400:, np.newaxis, 1:])
        o_data = np.loadtxt('./datasets/o_train', delimiter=',', skiprows=1)
        self.o = torch.from_numpy(o_data[1400:, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadTestSet(Dataset): 
    def __init__(self): 
        m_data = np.loadtxt('./datasets/m_test', delimiter=',', skiprows=1)
        self.m = torch.from_numpy(m_data[:, np.newaxis, 1:])
        s_data = np.loadtxt('./datasets/s_test', delimiter=',', skiprows=1)
        self.s = torch.from_numpy(s_data[:, np.newaxis, 1:])
        o_data = np.loadtxt('./datasets/o_test', delimiter=',', skiprows=1)
        self.o = torch.from_numpy(o_data[:, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

def create_parameters(N_m=8): 
    P = torch.zeros(N_m, N_m)
    for i in range(N_m): 
        if not i: 
            P += torch.diag(torch.Tensor([0.80]*N_m), i)
        elif i==1: 
            P += torch.diag(torch.Tensor([0.15]*(8-i)), i)
            P += torch.diag(torch.Tensor([0.15]*i), i-8)
        else: 
            P += torch.diag(torch.Tensor([1/120]*(8-i)), i)
            P += torch.diag(torch.Tensor([1/120]*i), i-8)

    A = torch.Tensor([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9])
    B = torch.Tensor([0., -2., 2., -4., 0., 2., -2., 4.])
    C = A.clone()
    D = B.clone()
    beta = torch.Tensor([1]*N_m)
    return P, A, B, C, D, beta

def generate_data(T, model, batch=1, dyn="Mark"): 
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

def weights_bootstrap(model, w_list, m_list, s_list, o): 
    lw = torch.log(w_list).to(device)
    o = torch.ones(s_list.size()).to(device) * o.to(device)
    s_obs = s_list.clone().detach()
    o -= (model.co_C[m_list].to(device) * torch.sqrt(torch.abs(s_obs)).to(device) + model.co_D[m_list].to(device))
    lw += torch.distributions.normal.Normal(loc=0., scale=model.sigma_v).log_prob(o)
#     for i in range(len(s_list)): 
#         lw[i] += stats.norm(loc=model.co_C[m_list[i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[i]], scale=np.sqrt(0.1)).logpdf(o)
# #         w += 10**(-300)
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True)

def weights_proposal(model, w_list, m_list, s_list, o, dyn): 
    lw = torch.log(w_list)
    o = torch.ones(s_list.size()) * o
    o -= (model.co_C[m_list[:, :, -1]] * torch.sqrt(torch.abs(s_list)) + model.co_D[m_list[:, :, -1]])
    if dyn=="Mark": 
        lp = torch.log(model.mat_P[m_list[:, :, -2], m_list[:, :, -1]])
    else: 
        batch = m_list.size()[0]
        N_p = m_list.size()[1]
        alpha = torch.zeros(batch, N_p, len(model.beta))
        for m_index in range(len(model.beta)): 
            alpha[:, :, m_index] = torch.sum(m_list[:, :, :-1] == m_index, dim=2)
        beta = model.beta[None, None, :]
        prob = (alpha + beta)/(alpha + beta).sum(dim=2, keepdim=True)
        batch_index = tuple(i//N_p for i in range(batch * N_p))
        pars_index = tuple([i for i in range(N_p)] * batch)
        index_flatten = tuple(m_list[:, :, -1].view(-1))
        lp = torch.log(prob[batch_index, pars_index, index_flatten].view(batch, N_p))

    lw += torch.distributions.normal.Normal(loc=0., scale=0.1**(0.5)).log_prob(o) + lp - torch.log((1/len(model.co_A)))
    # for i in range(len(s_list)): 
    #     lw[i] += stats.norm(loc=model.co_C[m_list[-1, i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[-1, i]], scale=np.sqrt(0.1)).logpdf(o) + lp[i] - np.log((1/len(model.co_A)))
#         w += 10**(-300)
#         print(lw[i])
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True)


def inv_cdf(cum, ws): 
    index = torch.ones(ws.size(), dtype=int).to(device) * cum.size()[-1]
    w_cum = ws.cumsum(axis=1).clone().detach().to(device)
    for i in range(cum.size()[-1]): 
        index[:, [i]] -= (cum[:, [i]].to(device) < w_cum).sum(dim=1, keepdim=True)
        # while cum[i] > w: 
        #     k += 1
        #     w += ws[k]
        # index[i] = k
    index = torch.where(index < ws.shape[-1], index, ws.shape[-1]-1)
    return index
    

def resample_systematic(ws): 
    batch, N_p = ws.size()
    cum = (torch.rand(batch, 1) + torch.arange(N_p).tile(batch, 1)) / torch.ones(batch, 1)*N_p   
    
    return inv_cdf(cum, ws)
    
def resample_multinomial(ws): 
    batch, N_p = ws.size()
    uni = (-torch.log(torch.rand(batch, N_p+1, dtype=torch.double))).cumsum(dim=1)
    cum = uni[:, :-1] / uni[:, [-1]]

    return inv_cdf(cum, ws)