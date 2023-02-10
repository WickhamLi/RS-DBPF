import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = torch.device('cpu')

class LoadTrainSet(Dataset): 
    def __init__(self): 
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
    o = torch.ones(s_list.size()) * o

    s_obs = s_list.clone().detach()
    o -= (model.co_C[m_list] * torch.sqrt(torch.abs(s_obs)) + model.co_D[m_list])
    lw += torch.distributions.normal.Normal(loc=0., scale=model.sigma_v).log_prob(o)
#     for i in range(len(s_list)): 
#         lw[i] += stats.norm(loc=model.co_C[m_list[i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[i]], scale=np.sqrt(0.1)).logpdf(o)
# #         w += 10**(-300)
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True)

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
    lw += (logden_dyn + logden_obs - logden_prop)
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
    index = torch.ones(ws.size(), dtype=int) * cum.size()[-1]
    w_cum = ws.cumsum(axis=1).clone().detach()
    for i in range(cum.size()[-1]): 
        index[:, [i]] -= (cum[:, [i]] < w_cum).sum(dim=1, keepdim=True)
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

