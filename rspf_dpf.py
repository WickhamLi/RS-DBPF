import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from utils import *
from NFs import *

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = torch.device('cpu')

class RSPF: 
    def __init__(self, A, B, C, D, mu_u=torch.tensor(0.), sigma_u=torch.tensor(0.1**(0.5)), mu_v=torch.tensor(0.), sigma_v=torch.tensor(0.1**(0.5)), tran_matrix=False, beta=False): 
        self.co_A = A
        self.co_B = B
        self.co_C = C
        self.co_D = D
        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.mu_v = mu_v
        self.sigma_v = sigma_v
        self.mat_P = tran_matrix
        self.beta = beta

    def initial(self, size=(1, 1)): 
        return torch.multinomial(torch.ones(size[0], len(self.co_A), device=device), replacement=True, num_samples=size[1]), torch.Tensor(size=size).uniform_(-0.5, 0.5).to(device)

    def Markov_dynamic(self, m_p): 
        batch, N_p = m_p.shape
        N_m = self.mat_P.shape[-1]
        sample = torch.rand(batch, N_p, 1)
        m_t = torch.ones(batch, N_p, 1, dtype=torch.long) * N_m
        cum = self.mat_P[m_p, :].cumsum(axis=2)
        m_t -= (sample < cum).sum(dim=2, keepdim=True)
        return m_t.view(batch, N_p)
    
    def Polyaurn_dynamic(self, m_p):
        batch, N_p = m_p.shape[:2]
        N_m = len(self.beta)
        sample = torch.rand(batch, N_p, 1, device=device)
        m_t = torch.ones(batch, N_p, 1, dtype=torch.long, device=device) * N_m
        alpha = torch.zeros(batch, N_p, N_m, device=device)
        for m_index in range(N_m): 
            alpha[:, :, m_index] = (m_p == m_index).sum(dim=2)
        beta = self.beta[None, None, :]
        prob = (alpha + beta)/(alpha + beta).sum(dim=2, keepdim=True)
        m_t -= (sample < prob.cumsum(dim=2)).sum(axis=2, keepdims=True)
        return m_t.view(batch, N_p)
    
    def propuni_dynamic(self, size=1): 
        return torch.multinomial(torch.ones(size[0], len(self.co_A), device=device), replacement=True, num_samples=size[1])
        
    def state(self, m_t, s_p): 
        return self.co_A[m_t] * s_p + self.co_B[m_t] + torch.randn(s_p.shape, device=device) * self.sigma_u + self.mu_u
        
    def obs(self, m_t, s_t): 
        return self.co_C[m_t] * torch.sqrt(torch.abs(s_t)) + self.co_D[m_t] + torch.normal(mean=self.mu_v, std=self.sigma_v, size=s_t.shape)

    def filtering(self, m, s, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=torch.long, device=device)
        s_list = torch.zeros(batch, N_p, T, device=device)
        w_list = torch.zeros(batch, N_p, T, device=device)
        m_list[:, :, 0], s_list[:, :, 0] = self.initial(size=(batch, N_p))
        # m_list[:, :, [0]] = m
        # s_list[:, :, [0]] = s
        m_list_re = m_list.clone().detach()
        s_list_re = s_list.clone().detach()
        w_list[:, :, 0] = 1/N_p
        w_list_re = w_list.clone().detach()
        for t in range(1, T): 
            if prop=="Boot": 
                if dyn=="Poly": 
                    m_list[:, :, t] = self.Polyaurn_dynamic(m_list_re[:, :, :t])
                elif dyn=="Mark": 
                    m_list[:, :, t] = self.Markov_dynamic(m_list_re[:, :, t-1])
            elif prop=="Uni": 
                m_list[:, :, t] = self.propuni_dynamic(size=(batch, N_p))
            elif prop=='Deter': 
                m_list[:, :, t] = torch.arange(len(self.co_A)).tile(batch, int(N_p/len(self.co_A)))
            
            m_list_re[:, :, t] = m_list[:, :, t].clone().detach()
            s_list[:, :, [t]] = self.state(m_list[:, :, [t]], s_list_re[:, :, [t-1]])         
            
            if prop=="Boot": 
                w_list[:, :, [t]] = weights_bootstrap(self, w_list_re[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
            else:
                if dyn=="Mark": 
                    w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, t-1:t+1], s_list[:, :,[t]], o[:, :, [t]], dyn)
                elif dyn=="Poly": 
                    w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, :t+1], s_list[:, :, [t]], o[:, :, [t]], dyn)
            w_list_re[:, :, t] = w_list[:, :, t].clone().detach()
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < s_list.shape[1]
            if ESS.sum(): 
                re_index = torch.where(ESS)[0]
                index = torch.arange(N_p, dtype=torch.long, device=device).tile(batch, 1)
                if re=="sys": 
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t])
                else: 
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t])
                w_list_re[re_index, :, t] = 1 / N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                index_flatten = index.view(-1)
                m_trans = m_list[batch_index, index_flatten, :t+1].clone().detach()
                m_list_re[:, :, :t+1] = m_trans.view(len(re_index), N_p, -1)
                s_trans = s_list[batch_index, index_flatten, :t+1].clone().detach()
                s_list_re[:, :, :t+1] = s_trans.view(len(re_index), N_p, -1)

        return m_list_re, s_list_re, w_list_re

    def test(self, test_data, N_p=2000, dyn="Mark", prop="Boot", re="mul"): 
        for m_test, s_test, o_test in test_data: 
            m_parlist, s_parlist, w_parlist = self.filtering(m_test[:, :, [0]], s_test[:, :, [0]], o_test.to(device), N_p=N_p, dyn=dyn, prop=prop, re=re)
            s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
            mse_pf = ((s_est - s_test.to(device))**2).detach().numpy()
            mse_pfdf = pd.DataFrame(np.squeeze(mse_pf))
            mse_pfdf.to_csv(f'./results/rspf/mse_{dyn}_{prop}_{re}')


class RSDPF(nn.Module, RSPF): 
    def __init__(self, rs=True, nf=False, mu_u=torch.tensor(0.), sigma_u=torch.Tensor(1).uniform_(0.1, 0.5), mu_v=torch.tensor(0.), sigma_v=torch.Tensor(1).uniform_(0.1, 0.5), tran_matrix=False, beta=False, learning_rate=1e-4): 
        super().__init__()
        self.mat_P = nn.Parameter(tran_matrix)
        self.beta = nn.Parameter(beta)
        if rs: 
            self.co_A = nn.Parameter(torch.Tensor(self.mat_P.shape[-1]).uniform_(-1, 1))
            self.co_B = nn.Parameter(torch.Tensor(self.mat_P.shape[-1]).uniform_(-4, 4))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        else: 
            self.co_A = nn.Parameter(torch.Tensor(1).uniform_(-1, 1))
            self.co_B = nn.Parameter(torch.Tensor(1).uniform_(-4, 4))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        self.mu_u = mu_u
        self.sigma_u = nn.Parameter(sigma_u)
        self.mu_v = mu_v
        self.sigma_v = nn.Parameter(sigma_v)
        if nf: 
            self.proposal = Cond_PlanarFlows(dim=1, N_m=8, layer=2)
            # self.measure = Cond_PlanarFlows(dim=1, N_m=8, layer=2)
        self.rs = rs
        self.nf = nf
        self.loss = nn.MSELoss()
        self.optim = torch.optim.SGD([self.co_A, self.co_B, self.co_C, self.co_D], lr = learning_rate, momentum=0.9)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10, 20, 30], gamma=0.8)

    def state(self, m_t, s_p): 
        s_t = self.co_A[m_t] * s_p + self.co_B[m_t] + torch.randn(s_p.shape, device=device) * self.sigma_u + self.mu_u
        return torch.where(s_t==0., s_t+1e-4, s_t)

    def filtering(self, m, s, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=torch.long)
        s_list = torch.zeros(batch, N_p, T)
        w_list = torch.zeros(batch, N_p, T)
        if len(self.co_A) == 1: 
            _, s_list[:, :, 0] = self.initial(size=(batch, N_p))
        else: 
            m_list[:, :, 0], s_list[:, :, 0] = self.initial(size=(batch, N_p))
        # m_list[:, :, [0]] = m
        # s_list[:, :, [0]] = s
        m_list_re = m_list.clone().detach()
        s_list_re = s_list.clone().detach()
        if self.nf: 
            s_list_prop = s_list.clone().detach()
            # z_list = s_list.clone().detach()
        w_list[:, :, 0] = 1/N_p
        w_list_re = w_list.clone().detach()
        for t in range(1, T): 
            if len(self.co_A) > 1: 
                if prop=="Boot": 
                    if dyn=="Poly": 
                        m_list[:, :, t] = self.Polyaurn_dynamic(m_list_re[:, :, :t])
                    elif dyn=="Mark": 
                        m_list[:, :, t] = self.Markov_dynamic(m_list_re[:, :, t-1])
                elif prop=="Uni": 
                    m_list[:, :, t] = self.propuni_dynamic(size=(batch, N_p))
                elif prop=='Deter': 
                    m_list[:, :, t] = torch.arange(len(self.co_A)).tile(batch, int(N_p/len(self.co_A)))
            
            m_list_re[:, :, t] = m_list[:, :, t].clone().detach()
            s_list[:, :, [t]] = self.state(m_list[:, :, [t]], s_list_re[:, :, [t-1]])         
            
            if self.nf: 
                s_list_prop[:, :, [t]], logdetJ_prop = self.proposal(m=m_list[:, :, [t]], s=s_list[:, :, [t]], o=o[:, :, [t]])
                # z_list[:, :, [t]], logdetJ_obs = self.measure(m=m_list[:, :, [t]], s=o[:, :, [t]], o=s_list_prop[:, :, [t]])
                logden_dyn = dyn_density(self, m_list[:, :, [t]], s_list[:, :, [t]], s_list_re[:, :, [t-1]])
                logden_prop = dyn_density(self, m_list[:, :, [t]], s_list_prop[:, :, [t]], s_list_re[:, :, [t-1]], logdetJ_prop)
                # logden_obs = obs_density(z_list[:, :, [t]], logdetJ_obs)

                if prop=="Boot": 
                    w_list[:, :, [t]] = weights_CNFs(w_list_re[:, :, [t-1]], logden_dyn, logden_prop, m_list[:, :, [t]], s_list_prop[:, :, [t]], o[:, :, [t]], self)
            else: 
                if prop=="Boot": 
                    w_list[:, :, [t]] = weights_bootstrap(self, w_list_re[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
                else:
                    if dyn=="Mark": 
                        w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, t-1:t+1], s_list[:, :, [t]], o[:, :, [t]], dyn)
                    else: 
                        w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, :t+1], s_list[:, :, [t]], o[:, :, [t]], dyn)
            w_list_re[:, :, t] = w_list[:, :, t].clone().detach()
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < s_list.shape[1]
    #         print(ESS)
            if ESS.sum(): 
                re_index = torch.where(ESS)[0]
                index = torch.arange(N_p, dtype=torch.long, device=device).tile(batch, 1)                
                if re=="sys": 
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t])
                else: 
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t])
                w_list_re[re_index, :, t] = 1 / N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                index_flatten = index.view(-1)
                if len(self.co_A) > 1: 
                    m_trans = m_list[batch_index, index_flatten, :t+1].clone().detach()
                    m_list_re[:, :, :t+1] = m_trans.view(len(re_index), N_p, -1)
                s_trans = s_list[batch_index, index_flatten, :t+1].clone().detach()
                s_list_re[:, :, :t+1] = s_trans.view(len(re_index), N_p, -1)

        return m_list, s_list, w_list, m_list_re, s_list_re, w_list_re

    def forward(self, m_data, s_data, o_data, N_p, dyn, prop, re, train=True):  
        m_parlist, s_parlist, w_parlist, m_re, s_re, w_re = self.filtering(m_data, s_data, o_data, N_p=N_p, dyn=dyn, prop=prop, re=re)
        if train: 
            s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
        else: 
            s_est = (w_re * s_re).sum(dim=1, keepdim=True)
        return s_est
    
    def train(self, train_data, val_data, N_iter=50, N_p=200, dyn="Mark", prop="Boot", re="mul"): 
        N_step = len(train_data)
        l = np.ones(N_iter) * 1e2
        for epoch in range(N_iter): 
            for i, (m_train, s_train, o_train) in enumerate(train_data): 
                s_est = self(m_train[:, :, [0]], s_train[:, :, [0]], o_train, N_p, dyn, prop, re)
                # loss_sample = ((s_est - s_data)**2).mean(dim=2, keepdim=True)
                nll = - torch.distributions.Normal(loc=s_est, scale=self.sigma_u).log_prob(s_train)
                loss = self.loss(s_est, s_train) + nll.mean()
                # loss_sample.mean(dim=0, keepdim=True)
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()
                print(f'epoch{epoch+1}/{N_iter}, step{i+1}/{N_step}: training loss = {loss.item():.8f}')
            self.optim_scheduler.step()
            with torch.no_grad(): 
                for m_val, s_val, o_val in val_data: 
                    s_est = self(m_val[:, :, [0]], s_val[:, :, [0]], o_val, N_p, dyn, prop, re, train=False)
                    # loss = ((s_est - s_val)**2).mean()
                    loss = self.loss(s_est, s_val)
                    if loss.item() < l.min(): 
                        torch.save(self, f'./results/rsdpf/bestval_rs{self.rs}_nf{self.nf}_{dyn}_{prop}_{re}')
                    l[epoch] = loss
            print(f'epoch{epoch+1}/{N_iter}: validation loss = {loss:.8f}')
            
        print(l)

    def test(self, test_data, N_p=2000, dyn="Mark", prop="Boot", re="mul"): 
        best_val = torch.load(f'./results/rsdpf/bestval_rs{self.rs}_nf{self.nf}_{dyn}_{prop}_{re}')
        with torch.no_grad(): 
            for m_test, s_test, o_test in test_data: 
                s_est = best_val(m_test[:, :, [0]], s_test[:, :, [0]], o_test, N_p, dyn, prop, re, train=False)
                loss = self.loss(s_est, s_test)
                print(f'DPF test loss = {loss.item():.8f}')
            mse_dpf = ((s_est - s_test)**2).detach().numpy()
            mse_dpfdf = pd.DataFrame(np.squeeze(mse_dpf))
            mse_dpfdf.to_csv(f'./results/rsdpf/mse_rs{self.rs}_nf{self.nf}_{dyn}_{prop}_{re}')


class MMPF(RSPF): 

    def __init__(self, A, B, C, D, mu_u=torch.tensor(0.), sigma_u=torch.tensor(0.1**(0.5)), mu_v=torch.tensor(0.), sigma_v=torch.tensor(0.1**(0.5)), gamma=torch.tensor(1.0)): 
        self.co_A = A.view(1, -1, 1, 1)
        self.co_B = B.view(1, -1, 1, 1)
        self.co_C = C.view(1, -1, 1, 1)
        self.co_D = D.view(1, -1, 1, 1)
        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.mu_v = mu_v
        self.sigma_v = sigma_v
        self.gamma = gamma

    def state(self, s_p): 
        return self.co_A * s_p + self.co_B + torch.randn(s_p.shape, device=device) * self.sigma_u + self.mu_u

    def filtering(self, s, o, N_p, re): 
        batch, _, T = o.shape
        N_m = self.co_A.shape[1]
        N_pm = int(N_p/N_m)
        s_list = torch.zeros(batch, N_m, N_pm, T, device=device)
        w_list = torch.zeros(batch, N_m, N_pm, T, device=device)   
        pi_list = torch.zeros(batch, N_m, 1, T, device=device)     
        _, s_list[:, :, :, 0] = self.initial(size=(batch, N_m, N_pm))
        # s_list[:, :, :, 0] = s
        s_list_re = s_list.clone().detach()
        w_list[:, :, :, 0] = N_m/N_p
        w_list_re = w_list.clone().detach()
        pi_list[:, :, :, 0] = 1/self.co_A.shape[1]
        for t in range(1, T): 
            s_list[:, :, :, [t]] = self.state(s_list_re[:, :, :, [t-1]])  
            pi_list[:, :, :, [t]] = pi_list[:, :, :, [t-1]]**self.gamma / (pi_list[:, :, :, [t-1]]**self.gamma).sum(dim=1, keepdim=True)
            w_list[:, :, :, [t]], pi_list[:, :, :, [t]] = weights_mmpf(self, w_list_re[:, :, :, [t-1]], s_list[:, :, :, [t]], pi_list[:, :, :, [t]], o[:, :, [t]])
            w_list_re[:, :, :, t] = w_list[:, :, :, t].clone().detach()
           
            ESS = 1 / (w_list[:, :, :, t]**2).sum(dim=2) < s_list.shape[2]
            if ESS.sum(): 
                re_index1, re_index2 = torch.where(ESS)[0], torch.where(ESS)[1]
                index = torch.arange(N_pm, dtype=torch.long, device=device).tile(batch, N_m, 1)                
                if re=="sys":                    
                    index[re_index1, re_index2, :] = resample_systematic_mmpf(w_list[re_index1, re_index2, :, t], len(torch.unique(re_index1)))
                else: 
                    index[re_index1, re_index2, :] = resample_multinomial_mmpf(w_list[re_index1, re_index2, :, t], len(torch.unique(re_index1)))
                w_list_re[re_index1, re_index2, :, t] = N_m/N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                m_index = torch.arange(N_m, dtype=torch.long).tile(N_pm, 1).T.reshape(-1).tile(batch)
                index_flatten = index.view(-1)
                s_trans = s_list[batch_index, m_index, index_flatten, :t+1].clone().detach()
                
                s_list_re[:, :, :, :t+1] = s_trans.view(batch, N_m, N_pm, -1)

        return pi_list, s_list_re, w_list_re

    def test(self, test_data, N_p=2000, re="mul", dyn="Mark"): 
        for m_test, s_test, o_test in test_data: 
            pi_list, s_parlist, w_parlist = self.filtering(s_test[:, :, [0]], o_test.to(device), N_p=N_p, re=re)
            s_est = (pi_list * (w_parlist * s_parlist).sum(dim=2, keepdim=True)).sum(dim=1)
            mse_mmpf = ((s_est - s_test.to(device))**2).detach().numpy()
            mse_mmpfdf = pd.DataFrame(np.squeeze(mse_mmpf))
            mse_mmpfdf.to_csv(f'./results/mmpf/mse_{dyn}_gamma{self.gamma:.1f}_{re}')


class MMPF2(RSPF): 
    def __init__(self, A, B, C, D, mu_u=torch.tensor(0.), sigma_u=torch.tensor(0.1**(0.5)), mu_v=torch.tensor(0.), sigma_v=torch.tensor(0.1**(0.5)), gamma=torch.tensor(1.0)): 
        self.co_A = A.view(1, -1, 1, 1)
        self.co_B = B.view(1, -1, 1, 1)
        self.co_C = C.view(1, -1, 1, 1)
        self.co_D = D.view(1, -1, 1, 1)
        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.mu_v = mu_v
        self.sigma_v = sigma_v
        self.gamma = gamma

    def state(self, s_p): 
        return self.co_A * s_p + self.co_B + torch.randn(s_p.shape, device=device) * self.sigma_u + self.mu_u

    def filtering(self, s, o, N_p, re): 
        batch, _, T = o.shape
        N_m = self.co_A.shape[1]
        N_pm = int(N_p/N_m)
        s_list = torch.zeros(batch, N_m, N_pm, T, device=device)
        w_list = torch.zeros(batch, N_m, N_pm, T, device=device)   
        pi_list = torch.zeros(batch, N_m, 1, T, device=device)     
        # _, s_list[:, :, :, 0] = self.initial(size=(batch, N_m, N_pm))
        s_list[:, :, :, 0] = s
        s_list_re = s_list.clone().detach()
        w_list[:, :, :, 0] = N_m/N_p
        w_list_re = w_list.clone().detach()
        pi_list[:, :, :, 0] = 1/self.co_A.shape[1]
        for t in range(1, T): 
            s_list[:, :, :, [t]] = self.state(s_list_re[:, :, :, [t-1]])  
            pi_list[:, :, :, [t]] = pi_list[:, :, :, [t-1]]**self.gamma / (pi_list[:, :, :, [t-1]]**self.gamma).sum(dim=1, keepdim=True)
            w_list[:, :, :, [t]], pi_list[:, :, :, [t]] = weights_mmpf(self, w_list_re[:, :, :, [t-1]], s_list[:, :, :, [t]], pi_list[:, :, :, [t]], o[:, :, [t]])
            w_list_re[:, :, :, t] = w_list[:, :, :, t].clone().detach()
           
            ESS = 1 / ((w_list[:, :, :, t]**2).sum(dim=2)).sum(dim=1) < (s_list.shape[2] * s_list.shape[1])
            if ESS.sum(): 
                re_index = torch.where(ESS)[0]
                index = torch.arange(N_pm, dtype=torch.long, device=device).tile(batch, 1)                
                if re=="sys":                    
                    index[re_index, :] = resample_systematic((pi_list[re_index, :, :, t] * w_list[re_index, :, :, t]).sum(dim=1))
                else: 
                    index[re_index, :] = resample_multinomial((pi_list[re_index, :, :, t] * w_list[re_index, :, :, t]).sum(dim=1))
                w_list_re[re_index, :, :, t] = N_m/N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                m_index = torch.arange(N_m, dtype=torch.long).tile(N_pm, 1).T.reshape(-1).tile(batch)
                index_flatten = index[:, None, :].tile(1, N_m, 1).view(-1)
                s_trans = s_list[batch_index, m_index, index_flatten, [t]].clone().detach()
                s_list_re[:, :, :, t] = s_trans.view(batch, N_m, -1)
                # s_list_re[:, :, :, t] = s_list_re[:, :, :, t].max(dim=1, keepdim=True)[0]

        return pi_list, s_list_re, w_list_re

    def test(self, test_data, N_p=2000, re="mul"): 
        for m_test, s_test, o_test in test_data: 
            pi_list, s_parlist, w_parlist = self.filtering(s_test[:, :, [0]], o_test.to(device), N_p=N_p, re=re)
            s_est = (pi_list * (w_parlist * s_parlist).sum(dim=2, keepdim=True)).sum(dim=1)
            mse_mmpf = ((s_est - s_test.to(device))**2).detach().numpy()
            mse_mmpfdf = pd.DataFrame(np.squeeze(mse_mmpf))
            mse_mmpfdf.to_csv(f'./results/mmpf/mse_gamma{self.gamma:.1f}_{re}')


