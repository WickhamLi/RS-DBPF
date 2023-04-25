import torch
import numpy as np
from torch.utils.data import Dataset


class LoadTrainSet(Dataset): 
    def __init__(self, dir, train_size): 
        m_data = np.loadtxt(f'./datasets/{dir}/m_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.m = torch.from_numpy(m_data[:train_size, np.newaxis, 1:])
        s_data = np.loadtxt(f'./datasets/{dir}/s_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.s = torch.from_numpy(s_data[:train_size, np.newaxis, 1:])
        o_data = np.loadtxt(f'./datasets/{dir}/o_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.o = torch.from_numpy(o_data[:train_size, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadValSet(Dataset): 
    def __init__(self, dir, train_size): 
        m_data = np.loadtxt(f'./datasets/{dir}/m_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.m = torch.from_numpy(m_data[train_size:, np.newaxis, 1:])
        s_data = np.loadtxt(f'./datasets/{dir}/s_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.s = torch.from_numpy(s_data[train_size:, np.newaxis, 1:])
        o_data = np.loadtxt(f'./datasets/{dir}/o_train', delimiter=',', skiprows=1, dtype=np.float32)
        self.o = torch.from_numpy(o_data[train_size:, np.newaxis, 1:])
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
    o = o.tile(1, s_list.shape[1], 1) - (model.co_C[m_list] * torch.sqrt(torch.abs(s_list)) + model.co_D[m_list])
    lw += torch.distributions.Normal(loc=0., scale=model.sigma_v.data).log_prob(o)
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True), torch.log(w.mean(dim=1))

def weights_bootstrap_sin(model, w_list, s_list, o): 
    lw = torch.log(w_list)
    o = o.tile(1, s_list.shape[1], 1) - (model.co_C * torch.sqrt(torch.abs(s_list)) + model.co_D)
    lw += torch.distributions.Normal(loc=0., scale=model.sigma_v.data).log_prob(o)
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True), torch.log(w.mean(dim=1))

def weights_bootstrap_nn(model, w_list, o_list, o): 
    lw = torch.log(w_list)
    o = o.tile(1, o_list.shape[1], 1) - o_list
    lw += torch.distributions.Normal(loc=0., scale=torch.abs(model.sigma_v.data)).log_prob(o)
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True), torch.log(w.mean(dim=1))

def weights_proposal_nn(model, w_list, m_list, o_list, o, dyn): 
    lw = torch.log(w_list)
    o = o.tile(1, o_list.shape[1], 1) - o_list
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
    lw += torch.distributions.Normal(loc=0., scale=abs(model.sigma_v.data)).log_prob(o) + lp - torch.log(torch.tensor(1/len(model.beta)))

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

    lw += torch.distributions.Normal(loc=0., scale=model.sigma_v.data).log_prob(o) + lp - torch.log(torch.tensor(1/len(model.co_A)))
    w = torch.exp(lw - lw.max(dim=1, keepdim=True)[0])
    return w/w.sum(dim=1, keepdim=True)


def weights_mmpf(model, w_list, s_list, pi_list, o): 
    lw = torch.log(w_list)
    o = o.tile(1, s_list.shape[1], s_list.shape[2])[:, :, :, None] - (model.co_C * torch.sqrt(torch.abs(s_list)) + model.co_D)
    lw += torch.distributions.Normal(loc=0., scale=model.sigma_v).log_prob(o)
    w = torch.exp(lw - lw.max(dim=2, keepdim=True)[0])
    pi_list *= w.sum(dim=2, keepdim=True) 
    return w/w.sum(dim=2, keepdim=True), pi_list/pi_list.sum(dim=1, keepdim=True)

def inv_cdf(cum, ws, device): 
    index = torch.ones(ws.shape, dtype=torch.long, device=device) * cum.shape[-1]
    if device == torch.device('mps'): 
        w_cum = ws.to(torch.device('cpu')).cumsum(dim=1).clone().detach().to(device)
    else: 
        w_cum = ws.cumsum(dim=1).clone().detach()
    for i in range(cum.shape[-1]): 
        index[:, [i]] -= (cum[:, [i]] < w_cum).sum(dim=1, keepdim=True, dtype=torch.long)
    index = torch.where(index < ws.shape[-1], index, ws.shape[-1]-1)
    return index

def inv_cdf_mmpf(cum, ws, device): 
    index = torch.ones(ws.shape, dtype=torch.long, device=device) * cum.shape[-1]
    if device == torch.device('mps'): 
        w_cum = ws.to(torch.device('cpu')).cumsum(dim=2).clone().detach().to(device)
    else: 
        w_cum = ws.cumsum(dim=2).clone().detach()
    for i in range(cum.shape[-1]): 
        index[:, :, [i]] -= (cum[:, :, [i]] < w_cum).sum(dim=2, keepdim=True, dtype=torch.long)
    index = torch.where(index < ws.shape[-1], index, ws.shape[-1]-1)
    return index.view(-1, ws.shape[-1])

def resample_systematic(ws, device): 
    batch, N_p = ws.shape
    cum = (torch.rand(batch, 1) + torch.arange(N_p).tile(batch, 1)) / torch.ones(batch, 1)*N_p   
    
    return inv_cdf(cum, ws, device)

def resample_systematic_mmpf(ws, batch, device): 
    N_p = ws.shape[-1]
    ws = ws.view(batch, -1, N_p)
    N_m = ws.shape[1]
    cum = (torch.rand(batch, N_m, 1) + torch.arange(N_p).tile(batch, N_m, 1)) / torch.ones(batch, N_m, 1)*N_p   
    return inv_cdf_mmpf(cum, ws, device)
    
def resample_multinomial(ws, device): 
    batch, N_p = ws.shape
    uni = (-torch.log(torch.rand(batch, N_p+1))).cumsum(dim=1).to(device)
    cum = uni[:, :-1] / uni[:, [-1]]

    return inv_cdf(cum, ws, device)

def resample_multinomial_mmpf(ws, batch, device): 
    N_p = ws.shape[-1]
    ws = ws.view(batch, -1, N_p)
    N_m = ws.shape[1]
    uni = (-torch.log(torch.rand(batch, N_m, N_p+1))).cumsum(dim=2).to(device)
    cum = uni[:, :, :-1] / uni[:, :, [-1]]

    return inv_cdf_mmpf(cum, ws, device)

