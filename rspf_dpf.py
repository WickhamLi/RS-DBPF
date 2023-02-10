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
        batch, N_p = m_p.size()
        N_m = self.mat_P.size()[-1]
        sample = torch.rand(batch, N_p, 1)
        m_t = torch.ones(batch, N_p, 1, dtype=int) * N_m
        cum = self.mat_P[m_p, :].cumsum(axis=2)
        m_t -= (sample < cum).sum(dim=2, keepdim=True)
        return m_t.view(batch, N_p)
    
    def Polyaurn_dynamic(self, m_p):
        batch, N_p = m_p.size()[:2]
        N_m = len(self.beta)
        sample = torch.rand(batch, N_p, 1, device=device)
        m_t = torch.ones(batch, N_p, 1, dtype=int, device=device) * N_m
        alpha = torch.zeros(batch, N_p, N_m, device=device)
        for m_index in range(N_m): 
            alpha[:, :, m_index] = (m_p == m_index).sum(dim=2)
        beta = self.beta[None, None, :]
        prob = (alpha + beta)/(alpha + beta).sum(dim=2, keepdim=True)
        m_t -= (sample < prob.cumsum(dim=2)).sum(axis=2, keepdims=True)
        return m_t.view(batch, N_p)
    
    def propuni_dynamic(self, size=1): 
        m_t = torch.multinomial(torch.Tensor([1]*len(self.co_A)), replacement=True, num_samples=size)
        return m_t
        
    def state(self, m_t, s_p): 
        return self.co_A[m_t] * s_p + self.co_B[m_t] + torch.randn(s_p.size(), device=device) * self.sigma_u + self.mu_u
        
    def obs(self, m_t, s_t): 
        return self.co_C[m_t] * torch.sqrt(torch.abs(s_t)) + self.co_D[m_t] + torch.normal(mean=self.mu_v, std=self.sigma_v, size=s_t.size())

    def filtering(self, m, s, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=int, device=device)
        s_list = torch.zeros(batch, N_p, T, device=device)
        w_list = torch.zeros(batch, N_p, T, device=device)
        m_list[:, :, 0], s_list[:, :, 0] = self.initial(size=(batch, N_p))
        # m_list[:, :, [0]] = m
        # s_list[:, :, [0]] = s
        w_list[:, :, 0] = 1/N_p
        for t in range(1, T): 
            if prop=="Boot": 
                if dyn=="Poly": 
                    m_list[:, :, t] = self.Polyaurn_dynamic(m_list[:, :, :t])
                elif dyn=="Mark": 
                    m_list[:, :, t] = self.Markov_dynamic(m_list[:, :, t-1])
            elif prop=="Uni": 
                m_list[:, :, t] = self.propuni_dynamic(size=(batch, N_p))
            elif prop=='Deter': 
                m_list[:, :, t] = torch.arange(len(self.co_A)).tile(batch, int(N_p/len(self.co_A)))
            
            s_list[:, :, [t]] = self.state(m_list[:, :, [t]], s_list[:, :, [t-1]])         
            
            if prop=="Boot": 
                w_list[:, :, [t]] = weights_bootstrap(self, w_list[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
            else:
                if dyn=="Mark": 
                    w_list[:, :, t] = weights_proposal(self, w_list[:, :, t-1], m_list[:, :, t-1:t+1], s_list[:, :, t], o[:, [t]], dyn)
                elif dyn=="Poly": 
                    w_list[:, :, t] = weights_proposal(self, w_list[:, :, t-1], m_list[:, :, :t+1], s_list[:, :, t], o[:, [t]], dyn)
            
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < s_list.size()[1]
            if ESS.sum(): 
                index = torch.arange(N_p, device=device).tile(batch, 1)
                if re=="sys": 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t])
                else: 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t])
                w_list[re_index, :, t] = 1 / N_p
                
                batch_index = tuple(i//N_p for i in range(batch * N_p))
                index_flatten = tuple(index.view(-1))
                m_trans = m_list[batch_index, index_flatten, [t]].clone().detach()
                m_list[:, :, t] = m_trans.view(batch, -1)
                s_trans = s_list[batch_index, index_flatten, [t]].clone().detach()
                s_list[:, :, t] = s_trans.view(batch, -1)

        return m_list, s_list, w_list

    def testing(self, test_data, N_p=2000, dyn="Mark", prop="Boot", re="mul"): 
        for m_test, s_test, o_test in test_data: 
            m_parlist, s_parlist, w_parlist = self.filtering(m_test[:, :, [0]], s_test[:, :, [0]], o_test.to(device), N_p=N_p, dyn=dyn, prop=prop, re=re)
            s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
            mse_pf = ((s_est - s_test.to(device))**2).detach().numpy()
            mse_pfdf = pd.DataFrame(np.squeeze(mse_pf))
            mse_pfdf.to_csv(f'./results/rspf/mse_{dyn}_{prop}_{re}')


class RSDPF(nn.Module, RSPF): 
    def __init__(self, rs=False, nf=False, sigma_u=torch.Tensor(1).uniform_(0.1, 0.5), sigma_v=torch.Tensor(1).uniform_(0.1, 0.5), tran_matrix=False, beta=False, learning_rate=1e-3): 
        super().__init__()
        self.mat_P = nn.Parameter(tran_matrix)
        self.beta = nn.Parameter(beta)
        if rs: 
            self.co_A = nn.Parameter(torch.Tensor(self.mat_P.size()[-1]).uniform_(-1, 1))
            self.co_B = nn.Parameter(torch.Tensor(self.mat_P.size()[-1]).uniform_(-4, 4))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        else: 
            self.co_A = nn.Parameter(torch.Tensor(1).uniform_(-1, 1))
            self.co_B = nn.Parameter(torch.Tensor(1).uniform_(-4, 4))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        self.sigma_u = nn.Parameter(sigma_u)
        self.sigma_v = nn.Parameter(sigma_v)
        if nf: 
            self.proposal = Cond_PlanarFlows(dim=1, N_m=8, layer=2)
            self.measure = Cond_PlanarFlows(dim=1, N_m=8, layer=2)
        self.rs = rs
        self.nf = nf
        self.loss = nn.MSELoss()
        self.optim = torch.optim.SGD([self.co_A, self.co_B, self.co_C, self.co_D], lr = learning_rate, momentum=0.9)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 10, 15, 25], gamma=0.5)

    def filtering(self, m, s, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=int)
        s_list = torch.zeros(batch, N_p, T)
        w_list = torch.zeros(batch, N_p, T)
        if self.co_A.dim() == 1: 
            _, s_list[:, :, 0] = self.initial(size=(batch, N_p))
        else: 
            m_list[:, :, 0], s_list[:, :, 0] = self.initial(size=(batch, N_p))
        # m_list[:, :, [0]] = m
        # s_list[:, :, [0]] = s
        m_list_re = m_list.clone().detach()
        s_list_re = s_list.clone().detach()
        if self.nf: 
            s_list_prop = s_list.clone().detach()
            z_list = s_list.clone().detach()
        w_list[:, :, 0] = 1/N_p
        w_list_re = w_list.clone().detach()
        for t in range(1, T): 
            if self.co_A.dim() > 1: 
                if prop=="Boot": 
                    if dyn=="Poly": 
                        m_list[:, :, t] = self.Polyaurn_dynamic(m_list_re[:, :, :t])
                    elif dyn=="Mark": 
                        m_list[:, :, t] = self.Markov_dynamic(m_list_re[:, :, t-1])
                elif prop=="Uni": 
                    m_list[:, :, t] = self.propuni_dynamic(size=(batch, N_p))
                elif prop=='Deter': 
                    m_list[:, :, t] = torch.arange(len(self.co_A)).tile(batch, int(N_p/len(self.co_A)))
            s_list[:, :, [t]] = self.state(m_list[:, :, [t]], s_list_re[:, :, [t-1]])         
            
            if self.nf: 
                s_list_prop[:, :, [t]], logdetJ_prop = self.proposal(m=m_list[:, :, [t]], s=s_list[:, :, [t]], o=o[:, :, [t]])
                z_list[:, :, [t]], logdetJ_obs = self.measure(m=m_list[:, :, [t]], s=o[:, :, [t]], o=s_list[:, :, [t]])
                logden_dyn = dyn_density(self, m_list[:, :, [t]], s_list[:, :, [t]], s_list_re[:, :, [t-1]])
                logden_prop = dyn_density(self, m_list[:, :, [t]], s_list_prop[:, :, [t]], s_list_re[:, :, [t-1]], logdetJ_prop)
                logden_obs = obs_density(z_list[:, :, [t]], logdetJ_obs)

                if prop=="Boot": 
                    w_list[:, :, [t]] = weights_CNFs(w_list_re[:, :, [t-1]], logden_dyn, logden_prop, logden_obs)
            else: 
                if prop=="Boot": 
                    w_list[:, :, [t]] = weights_bootstrap(self, w_list_re[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
                else:
                    if dyn=="Mark": 
                        w_list[:, :, t] = weights_proposal(self, w_list_re[:, :, t-1], m_list[:, :, t-1:t+1], s_list[:, :, t], o[:, [t]], dyn)
                    else: 
                        w_list[:, :, t] = weights_proposal(self, w_list_re[:, :, t-1], m_list[:, :, :t+1], s_list[:, :, t], o[:, [t]], dyn)
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < s_list.size()[1]
    #         print(ESS)
            if ESS.sum(): 
                index = torch.arange(N_p).tile(batch, 1)
                if re=="sys": 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t])
                else: 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t])
                w_list_re[re_index, :, t] = 1 / N_p
                
                batch_index = tuple(i//N_p for i in range(batch * N_p))
                index_flatten = tuple(index.view(-1))
                if self.co_A.dim() == 8: 
                    m_trans = m_list[batch_index, index_flatten, [t]].clone().detach()
                    m_list_re[:, :, t] = m_trans.view(batch, -1)
                s_trans = s_list[batch_index, index_flatten, [t]].clone().detach()
                s_list_re[:, :, t] = s_trans.view(batch, -1)

        return m_list, s_list, w_list

    def forward(self, m_data, s_data, o_data, N_p, dyn, prop, re, nf):  
        m_parlist, s_parlist, w_parlist = self.filtering(self, m_data, s_data, o_data, N_p=N_p, dyn=dyn, prop=prop, re=re)
        s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
        return s_est
    
    def training(self, train_data, val_data, N_iter=40, N_p=200, dyn="Mark", prop="Boot", re="mul"): 
        N_step = len(train_data)
        l = np.ones(N_iter) * 1e2
        for epoch in range(N_iter): 
            for i, (m_train, s_train, o_train) in enumerate(train_data): 
                s_est = self(m_train[:, :, [0]], s_train[:, :, [0]], o_train, N_p, dyn, prop, re)
                # loss_sample = ((s_est - s_data)**2).mean(dim=2, keepdim=True)
                loss = self.loss(s_est, s_train)
                # loss_sample.mean(dim=0, keepdim=True)
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()
                print(f'epoch{epoch+1}/{N_iter}, step{i+1}/{N_step}: training loss = {loss.item():.8f}')
            self.optim_scheduler.step()
            with torch.no_grad(): 
                for m_val, s_val, o_val in val_data: 
                    s_est = self(m_val[:, :, [0]], s_val[:, :, [0]], o_val, N_p, dyn, prop, re)
                    # loss = ((s_est - s_val)**2).mean()
                    loss = self.loss(s_est, s_val)
                    if loss.item() < l.min(): 
                        torch.save(self, f'./results/best_val_{dyn}_{prop}_{re}')
                    l[epoch] = loss
            print(f'epoch{epoch+1}/{N_iter}: validation loss = {loss:.8f}')
            
        print(l)

    def testing(self, test_data, N_p=2000, dyn="Mark", prop="Boot", re="mul"): 
        best_val = torch.load(f'./results/best_val_{dyn}_{prop}_{re}')
        with torch.no_grad(): 
            for m_test, s_test, o_test in test_data: 
                s_est = best_val(m_test[:, :, [0]], s_test[:, :, [0]], o_test, N_p, dyn, prop, re)
                loss = self.loss(s_est, s_test)
                print(f'DPF test loss = {loss.item():.8f}')
            mse_dpf = ((s_est - s_test)**2).detach().numpy()
            mse_dpfdf = pd.DataFrame(np.squeeze(mse_dpf))
            mse_dpfdf.to_csv(f'./results/rsdpf_{self.rs}/mse_{dyn}_{prop}_{re}')


class MMPF(RSPF): 

    def __init__(self, A, B, C, D, mu_u=torch.tensor(0.), sigma_u=torch.tensor(0.1**(0.5)), mu_v=torch.tensor(0.), sigma_v=torch.tensor(0.1**(0.5))): 
        self.co_A = A
        self.co_B = B
        self.co_C = C
        self.co_D = D
        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.mu_v = mu_v
        self.sigma_v = sigma_v

    def model_trans(self, m_p, gamma=1.): 
        m_p**gamma / (m_p**gamma).sum(dim=1, keepdim=True)

    def filtering(self, m, s, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape       
        m_list = torch.tensor(i//(N_p/len(self.co_A)) for i in range(N_p)).view(1, -1, 1)
        s_list = torch.zeros(batch, N_p, T, device=device)
        w_list = torch.zeros(batch, N_p, T, device=device)   
        pi_list = torch.zeros(batch, len(self.co_A), T, device=device)     
        _, s_list[:, :, 0] = self.initial(size=(batch, N_p))
        # m_list[:, :, [0]] = m
        # s_list[:, :, [0]] = s
        w_list[:, :, 0] = 1/N_p
        pi_list[:, :, 0] = 1/len(self.co_A)
        for t in range(1, T): 
            s_list[:, :, [t]] = self.state(m_list, s_list[:, :, [t-1]])  

            # m_list[:, :, t] = self.Polyaurn_dynamic(m_list[:, :, :t])
            # m_list[:, :, t] = self.Markov_dynamic(m_list[:, :, t-1])
            # m_list[:, :, t] = self.propuni_dynamic(size=(batch, N_p))
            # m_list[:, :, t] = torch.arange(len(self.co_A)).tile(batch, int(N_p/len(self.co_A)))
            
            if prop=="Boot": 
                w_list[:, :, [t]] = weights_bootstrap(self, w_list[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
            else:
                if dyn=="Mark": 
                    w_list[:, :, t] = weights_proposal(self, w_list[:, :, t-1], m_list[:, :, t-1:t+1], s_list[:, :, t], o[:, [t]], dyn)
                elif dyn=="Poly": 
                    w_list[:, :, t] = weights_proposal(self, w_list[:, :, t-1], m_list[:, :, :t+1], s_list[:, :, t], o[:, [t]], dyn)
            
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < s_list.size()[1]
            if ESS.sum(): 
                index = torch.arange(N_p, device=device).tile(batch, 1)
                if re=="sys": 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t])
                else: 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t])
                w_list[re_index, :, t] = 1 / N_p
                
                batch_index = tuple(i//N_p for i in range(batch * N_p))
                index_flatten = tuple(index.view(-1))
                m_trans = m_list[batch_index, index_flatten, [t]].clone().detach()
                m_list[:, :, t] = m_trans.view(batch, -1)
                s_trans = s_list[batch_index, index_flatten, [t]].clone().detach()
                s_list[:, :, t] = s_trans.view(batch, -1)

        return m_list, s_list, w_list


