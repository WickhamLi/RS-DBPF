import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from utils import *
from NFs import *

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
device = torch.device('cpu')

class RSDPF(nn.Module): 
        
    def __init__(self, tran_matrix, beta=torch.Tensor([1]), learning_rate=1e-3, rs=False): 
        super().__init__()
        self.mat_P = nn.Parameter(tran_matrix)
        self.beta = nn.Parameter(beta)
        if rs: 
            self.co_A = nn.Parameter(torch.Tensor(self.mat_P.size()[-1]).uniform_(-1, 1))
            self.co_B = nn.Parameter(torch.Tensor(self.mat_P.size()[-1]).uniform_(-4, 4))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        else: 
            self.co_A = nn.Parameter(torch.Tensor(1).uniform_(-1, 1).tile(self.mat_P.size()[-1]))
            self.co_B = nn.Parameter(torch.Tensor(1).uniform_(-4, 4).tile(self.mat_P.size()[-1]))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        self.sigma_u = nn.Parameter(torch.Tensor(1).uniform_(0.1, 0.5))
        self.sigma_v = nn.Parameter(torch.Tensor(1).uniform_(0.1, 0.5))
        self.proposal = Cond_PlanarFlows(dim=1, N_m=8, layer=2).to(device)
        self.measure = Cond_PlanarFlows(dim=1, N_m=8, layer=2).to(device)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.SGD([self.co_A, self.co_B, self.co_C, self.co_D], lr = learning_rate, momentum=0.9)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 10, 15, 25], gamma=0.5)

    def filtering(self, model, m, s, o, N_p, dyn, prop, re, nf): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=int).to(device)
        s_list = torch.zeros(batch, N_p, T).to(device)
        w_list = torch.zeros(batch, N_p, T).to(device)
        m_list[:, :, 0], s_list[:, :, 0] = model.initial(size=(batch, N_p))
        # m_list[:, :, [0]] = m
        # s_list[:, :, [0]] = s
        m_list_re = m_list.clone().detach()
        s_list_re = s_list.clone().detach()
        # s_list_proto = s_list.clone().detach()
        s_list_dyn = s_list.clone().detach()
        z_list = s_list.clone().detach()
        w_list[:, :, 0] = 1/N_p
        w_list_re = w_list.clone().detach()
        for t in range(1, T): 
            if prop=="Boot": 
                if dyn=="Poly": 
                    m_list[:, :, t] = model.Polyaurn_dynamic(m_list_re[:, :, :t])
                elif dyn=="Mark": 
                    m_list[:, :, t] = model.Markov_dynamic(m_list_re[:, :, t-1])
            elif prop=="Uni": 
                m_list[:, :, t] = model.propuni_dynamic(size=(batch, N_p))
            elif prop=='Deter': 
                m_list[:, :, t] = torch.arange(len(model.co_A)).tile(batch, int(N_p/len(model.co_A)))
            s_list_dyn[:, :, [t]] = model.state(m_list[:, :, [t]], s_list_re[:, :, [t-1]])
            # s_list_dyn[:, :, [t]], logdetJ_dyn = self.dynamic(m_list[:, :, [t]], s_list_proto[:, :, [t]])
            s_list[:, :, [t]], logdetJ_prop = self.proposal(m=m_list[:, :, [t]], s=s_list_dyn[:, :, [t]], o=o[:, :, [t]])
            z_list[:, :, [t]], logdetJ_obs = self.measure(m=m_list[:, :, [t]], s=o[:, :, [t]], o=s_list[:, :, [t]])
            
            if nf: 
                logden_dyn = dyn_density(model, m_list[:, :, [t]], s_list_dyn[:, :, [t]], s_list_re[:, :, [t-1]])
                logden_prop = dyn_density(model, m_list[:, :, [t]], s_list[:, :, [t]], s_list_re[:, :, [t-1]], logdetJ_prop)
                logden_obs = obs_density(z_list[:, :, [t]], logdetJ_obs)

                if prop=="Boot": 
                    w_list[:, :, [t]] = weights_CNFs(w_list_re[:, :, [t-1]], logden_dyn, logden_prop, logden_obs)
            else: 
                if prop=="Boot": 
                    w_list[:, :, [t]] = weights_bootstrap(model, w_list_re[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
                else:
                    if dyn=="Mark": 
                        w_list[:, :, t] = weights_proposal(model, w_list_re[:, :, t-1], m_list[:, :, t-1:t+1], s_list[:, :, t], o[:, [t]], dyn)
                    else: 
                        w_list[:, :, t] = weights_proposal(model, w_list_re[:, :, t-1], m_list[:, :, :t+1], s_list[:, :, t], o[:, [t]], dyn)
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < s_list.size()[1]
    #         print(ESS)
            if ESS.sum(): 
                index = torch.arange(N_p).tile(batch, 1).to(device)
                if re=="sys": 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t])
                else: 
                    re_index = torch.where(ESS)[0]
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t])
                w_list_re[re_index, :, t] = 1 / N_p
                # for b in range(batch): 
                #     for n in range(N_p): 
                #         m_list[b, n, t] = np.copy(m_list[b, index[b, n], t])
                #         s_list[b, n, t] = np.copy(m_list[b, index[b, n], t])
                
                batch_index = tuple(i//N_p for i in range(batch * N_p))
                index_flatten = tuple(index.view(-1))
                m_trans = m_list[batch_index, index_flatten, [t]].clone().detach()
                m_list_re[:, :, t] = m_trans.view(batch, -1)
                s_trans = s_list[batch_index, index_flatten, [t]].clone().detach()
                s_list_re[:, :, t] = s_trans.view(batch, -1)

        return m_list, s_list, w_list


    def forward(self, m_data, s_data, o_data, N_p, dyn, prop, re, nf):  
        model = RSPF(self.mat_P, self.co_A, self.co_B, self.co_C, self.co_D, sigma_u=self.sigma_u, sigma_v=self.sigma_v, beta=self.beta)
        m_parlist, s_parlist, w_parlist = self.filtering(model, m_data, s_data, o_data, N_p=N_p, dyn=dyn, prop=prop, re=re, nf=nf)
        s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
        return s_est


def training(model, train_data, val_data, N_iter=50, N_p=2000, dyn="Mark", prop="Boot", re="mul", nf=0): 
    N_step = len(train_data)
    l = np.ones(N_iter) * 1e2
    for epoch in range(N_iter): 
        for i, (m_data, s_data, o_data) in enumerate(train_data): 
            s_est = model(m_data[:, :, [0]], s_data[:, :, [0]], o_data.to(device), N_p, dyn, prop, re, nf=nf)
            loss_sample = ((s_est - s_data.to(device))**2).mean(dim=2, keepdim=True)
            loss = loss_sample.mean(dim=0, keepdim=True)
            # loss.requires_grad_(True)
            loss.backward()

            model.optim.step()
            model.optim.zero_grad()
            print(f'epoch{epoch+1}/{N_iter}, step{i+1}/{N_step}: training loss = {loss.item():.8f}')
        model.optim_scheduler.step()
        with torch.no_grad(): 
            for m_val, s_val, o_val in val_data: 
                s_est = model(m_val[:, :, [0]], s_val[:, :, [0]], o_val, N_p, dyn, prop, re, nf=nf)
                loss = ((s_est - s_val.to(device))**2).mean()
                if loss.item() < l.min(): 
                    torch.save(model, './results/best_val')
                l[epoch] = loss
        
        print(f'epoch{epoch+1}/{N_iter}: validation loss = {loss:.8f}')
        
    plt.plot(l, label='Loss Gradient Descent')
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/loss{dyn}{prop}{re}.png')
    plt.show()
    return l

def testing(model, test_data, A, B, C, D, N_p=2000, dyn="Mark", prop="Boot", re="mul", nf=0): 
    rsdpf = torch.load('./results/best_val')
    rspf = RSPF(model.mat_P, A, B, C, D, beta=model.beta)
    with torch.no_grad(): 
        for m_test, s_test, o_test in test_data: 
            s_estdpf = rsdpf.forward(m_test[:, :, [0]], s_test[:, :, [0]], o_test, N_p, dyn, prop, re, nf=nf)
            loss_dpf = ((s_estdpf - s_test.to(device))**2).mean()
            print(f'DPF test loss = {loss_dpf.item():.8f}')
            mse_dpf = ((s_estdpf - s_test)**2).detach().numpy()
            msecum_dpf = mse_dpf.cumsum(axis=-1)

            m_parlist, s_parlist, w_parlist = model.filtering(rspf, m_test[:, :, [0]], s_test[:, :, [0]], o_test, N_p=N_p, dyn=dyn, prop=prop, re=re, nf=0)
            s_estpf = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
            loss = ((s_estpf - s_test.to(device))**2).mean()
            print(f'PF test loss = {loss:.8f}')
            mse_pf = ((s_estpf - s_test)**2).detach().numpy()
            msecum_pf = mse_pf.cumsum(axis=-1)

            plt.cla()
            plt.plot(msecum_dpf.mean(axis=0)[-1], label=f'RSDPF({dyn}/{prop}/{re})')
            plt.plot(msecum_pf.mean(axis=0)[-1], label=f'RSPF({dyn}/{prop}/{re})')
            plt.ylabel('Average Cumulative MSE')
            plt.xlabel('Time Step')
            plt.yscale('log')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'./figures/Comparison({dyn}{prop}{re})')
            plt.show()

    return np.squeeze(mse_dpf), np.squeeze(mse_pf)


class RSPF: 
    def __init__(self, tran_matrix, A, B, C, D, mu_u=torch.tensor(0.), sigma_u=torch.tensor(0.1**(0.5)), mu_v=torch.tensor(0.), sigma_v=torch.tensor(0.1**(0.5)), beta=torch.tensor(1.)): 
        self.mat_P = tran_matrix
        self.co_A = A
        self.co_B = B
        self.co_C = C
        self.co_D = D
        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.mu_v = mu_v
        self.sigma_v = sigma_v
        self.beta = beta

    def initial(self, size=(1, 1)): 
        return torch.multinomial(torch.ones(size[0], len(self.co_A)), replacement=True, num_samples=size[1]), torch.Tensor(size=size).uniform_(-0.5, 0.5)

    def Markov_dynamic(self, m_p): 
        batch, N_p = m_p.size()
        N_m = self.mat_P.size()[-1]
        sample = torch.rand(batch, N_p, 1).to(device)
        m_t = torch.ones(batch, N_p, 1, dtype=int).to(device) * N_m
        cum = self.mat_P[m_p, :].cumsum(axis=2).to(device)
        m_t -= (sample < cum).sum(dim=2, keepdim=True)
        return m_t.view(batch, N_p)
    
    def Polyaurn_dynamic(self, m_p):
        batch, N_p = m_p.size()[:2]
        N_m = len(self.beta)
        sample = torch.rand(batch, N_p, 1)
        m_t = torch.ones(batch, N_p, 1, dtype=int) * N_m
        alpha = torch.zeros(batch, N_p, N_m)
        for m_index in range(N_m): 
            alpha[:, :, m_index] = (m_p == m_index).sum(dim=2)
        beta = self.beta[None, None, :]
        prob = (alpha + beta)/(alpha + beta).sum(dim=2, keepdim=True)
        cum = prob.cumsum(dim=2)
        m_t -= (sample < cum).sum(axis=2, keepdims=True)
        return m_t.view(batch, N_p)
    
    def propuni_dynamic(self, size=1): 
        m_t = torch.multinomial(torch.Tensor([1]*len(self.co_A)), replacement=True, num_samples=size)
        return m_t
        
    def state(self, m_t, s_p): 
        return self.co_A[m_t].to(device) * s_p.to(device) + self.co_B[m_t].to(device) + torch.normal(mean=self.mu_u, std=1., size=s_p.size()).to(device) * self.sigma_u.to(device)
        
    def obs(self, m_t, s_t): 
        return self.co_C[m_t] * torch.sqrt(torch.abs(s_t)) + self.co_D[m_t] + torch.normal(mean=self.mu_v, std=self.sigma_v, size=s_t.size())




