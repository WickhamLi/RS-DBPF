import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy import stats

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

class RSDPF(): 
    def __init__(self, tran_matrix, beta=torch.Tensor([1]), learning_rate=0.001): 
        self.mat_P = tran_matrix
        self.beta = beta
        self.co_A = torch.Tensor(self.mat_P.size()[-1]).uniform_(-1, 1)
        self.co_A.requires_grad_(True)
        self.co_B = torch.Tensor(self.mat_P.size()[-1]).uniform_(-4, 4)
        self.co_B.requires_grad_(True)
        self.co_C = self.co_A.clone().detach()
        self.co_C.requires_grad_(True)
        self.co_D = self.co_B.clone().detach()
        self.co_D.requires_grad_(True)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam([self.co_A, self.co_B, self.co_C, self.co_D], lr = learning_rate)

    def filtering(self, model, m, s, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=int)
        s_list = torch.zeros(batch, N_p, T)
        w_list = torch.zeros(batch, N_p, T)
        m_list[:, :, 0], s_list[:, :, 0] = model.initial(size=(batch, N_p))
        # m_list[:, :, [0]] = m
        # s_list[:, :, [0]] = s
        m_list_re = m_list.clone().detach()
        s_list_re = s_list.clone().detach()
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
            s_list[:, :, [t]] = model.state(m_list[:, :, [t]], s_list_re[:, :, [t-1]])
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
                index = torch.arange(N_p).tile(batch, 1)
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


    def forward(self, m_data, s_data, o_data, N_p, dyn, prop, re):  
        model = RSPF(self.mat_P, self.co_A, self.co_B, self.co_C, self.co_D, beta=self.beta)
        m_parlist, s_parlist, w_parlist = self.filtering(model, m_data, s_data, o_data, N_p=N_p, dyn=dyn, prop=prop, re=re)
        s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
        return s_est


    def training(self, m_data, s_data, o_data, N_iter=50, N_p=200, dyn="Mark", prop="Boot", re="mul"): 
        # s_data, o_data = data
        l = np.zeros(N_iter)
        for epoch in range(N_iter): 
            s_est = self.forward(m_data, s_data, o_data, N_p, dyn, prop, re)
            loss = ((s_est - s_data)**2).mean()
            loss.requires_grad_(True)
            loss.backward()
            l[epoch] = loss

            self.optim.step()
            self.optim.zero_grad()
        
            print(f'epoch{epoch+1}: loss = {loss:.8f}')
        plt.plot(l)
        plt.show()

    def testing(self, m_data, s_data, o_data, N_p=200, dyn="Mark", prop="Boot", re="mul"): 
        s_est = self.forward(m_data, s_data, o_data, N_p, dyn, prop, re)
        loss = self.loss(s_est, s_data)
        print(f'loss = {loss:.8f}')
        mse = ((s_est - s_data)**2).detach().numpy()
        mse_cum = mse.cumsum(axis=-1)
        plt.plot(mse_cum.mean(axis=0)[-1], label='RSPF(Bootstrap)')
        plt.ylabel('Average Cumulative MSE')
        plt.xlabel('Time Step')
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        # plt.savefig('RSPFMark.png')
        plt.show()


        print(mse_cum.mean(axis=0)[-1][-1])
        print(mse.mean(), mse.max(), mse.min())


        

# def MSE(s_list, s, w_list): 
#     s_est = (w_list * s_list).sum(dim=1)
#     mse = (s_est - s)**2
#     mse_cum = mse.cumsum(dim=-1)
#     return mse, mse_cum


class RSPF: 
    def __init__(self, tran_matrix, A, B, C, D, mu_u=0., sigma_u=0.1**(0.5), mu_v=0., sigma_v=0.1**(0.5), beta=1): 
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
        sample = torch.rand(batch, N_p, 1)
        m_t = torch.ones(batch, N_p, 1, dtype=int) * N_m
        cum = self.mat_P[m_p, :].cumsum(axis=2)
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
        x = self.co_A[m_t] * s_p + self.co_B[m_t] + torch.normal(mean=self.mu_u, std=self.sigma_u, size=s_p.size())
        return x
        
    def obs(self, m_t, s_t): 
        return self.co_C[m_t] * torch.sqrt(torch.abs(s_t)) + self.co_D[m_t] + torch.normal(mean=self.mu_v, std=self.sigma_v, size=s_t.size())
        

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
    lw = torch.log(w_list)
    o = torch.ones(s_list.size()) * o
    o -= (model.co_C[m_list] * torch.sqrt(torch.abs(s_list)) + model.co_D[m_list])
    lw += torch.distributions.normal.Normal(loc=0., scale=0.1**(0.5)).log_prob(o)
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



