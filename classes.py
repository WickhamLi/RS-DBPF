import pandas as pd
from utils import *
from NNs import *
import os


class RSPF: 
    def __init__(self, N_m, mu_u, sigma_u, mu_v, sigma_v, args, device): 
        self.N_m = N_m
        self.mat_P, self.co_A, self.co_B, self.co_C, self.co_D, self.beta = args
        self.mu_u = torch.tensor(mu_u)
        self.sigma_u = torch.tensor(sigma_u)
        self.mu_v = torch.tensor(mu_v)
        self.sigma_v = torch.tensor(sigma_v)
        self.device=device

    def initial(self, size=(1, 1)): 
        return torch.multinomial(torch.ones(size[0], self.N_m), replacement=True, num_samples=size[1]), torch.rand(size=size)-0.5

    def Markov_dynamic(self, m_p): 
        batch, N_p = m_p.shape
        sample = torch.rand(batch, N_p, 1, device=self.device)
        m_t = torch.ones(batch, N_p, 1, dtype=torch.long, device=self.device) * self.N_m

        # The operator 'aten::cumsum.out' is not currently implemented for the MPS device.
        if self.device == torch.device('mps'): 
            m_t -= (sample < self.mat_P[m_p, :].to(torch.device('cpu')).cumsum(dim=2).to(self.device)).sum(dim=2, keepdim=True)
        else: 
            m_t -= (sample < self.mat_P[m_p, :].cumsum(dim=2)).sum(dim=2, keepdim=True)

        return m_t.view(batch, N_p)
    
    def Polyaurn_dynamic(self, m_p):
        batch, N_p = m_p.shape[:2]
        sample = torch.rand(batch, N_p, 1, device=self.device)
        m_t = torch.ones(batch, N_p, 1, dtype=torch.long, device=self.device) * self.N_m
        alpha = torch.zeros(batch, N_p, self.N_m, device=self.device)
        for m_index in range(self.N_m): 
            alpha[:, :, m_index] = (m_p == m_index).sum(dim=2)
        beta = self.beta[None, None, :]
        prob = (alpha + beta)/(alpha + beta).sum(dim=2, keepdim=True)

        # The operator 'aten::cumsum.out' is not currently implemented for the MPS device.
        if self.device == torch.device('mps'): 
            m_t -= (sample < prob.to(torch.device('cpu')).cumsum(dim=2).to(self.device)).sum(dim=2, keepdims=True)
        else: 
            m_t -= (sample < prob.cumsum(dim=2)).sum(dim=2, keepdims=True)

        return m_t.view(batch, N_p)
    
    def propuni_dynamic(self, size=1): 
        return torch.multinomial(torch.ones(size[0], self.N_m, device=self.device), replacement=True, num_samples=size[1])
        
    def state(self, m_t, s_p): 
        return self.co_A[m_t] * s_p + self.co_B[m_t] + torch.normal(mean=self.mu_u, std=self.sigma_u, size=s_p.shape, device=self.device)
        
    def obs(self, m_t, s_t): 
        return self.co_C[m_t] * torch.sqrt(torch.abs(s_t)) + self.co_D[m_t] + torch.normal(mean=self.mu_v, std=self.sigma_v, size=s_t.shape, device=self.device)

    def filtering(self, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=torch.long, device=self.device)
        s_list = torch.zeros(batch, N_p, T, device=self.device)
        w_list = torch.zeros(batch, N_p, T, device=self.device)
        m_list[:, :, 0], s_list[:, :, 0] = self.initial(size=(batch, N_p))
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
                m_list[:, :, t] = torch.arange(self.N_m).tile(batch, int(N_p/self.N_m))
            
            m_list_re[:, :, t] = m_list[:, :, t].clone().detach()
            s_list[:, :, [t]] = self.state(m_list[:, :, [t]], s_list_re[:, :, [t-1]])         
            
            if prop=="Boot": 
                w_list[:, :, [t]], _ = weights_bootstrap(self, w_list_re[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
            else:
                if dyn=="Mark": 
                    w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, t-1:t+1], s_list[:, :,[t]], o[:, :, [t]], dyn)
                elif dyn=="Poly": 
                    w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, :t+1], s_list[:, :, [t]], o[:, :, [t]], dyn)
            w_list_re[:, :, t] = w_list[:, :, t].clone().detach()
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < (s_list.shape[1] + 1)
            if ESS.sum(): 
                re_index = torch.where(ESS)[0]
                index = torch.arange(N_p, dtype=torch.long, device=self.device).tile(batch, 1)
                if re=="sys": 
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t], device=self.device)
                else: 
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t], device=self.device)
                w_list_re[re_index, :, t] = 1 / N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                index_flatten = index.view(-1)
                m_trans = m_list[batch_index, index_flatten, :t+1].clone().detach()
                m_list_re[:, :, :t+1] = m_trans.view(len(re_index), N_p, -1)
                s_trans = s_list[batch_index, index_flatten, :t+1].clone().detach()
                s_list_re[:, :, :t+1] = s_trans.view(len(re_index), N_p, -1)

        return m_list, s_list, w_list

    def test(self, test_data, N_p=2000, dyn="Mark", prop="Boot", re="mul"): 
        for _, s_test, o_test in test_data: 
            m_parlist, s_parlist, w_parlist = self.filtering(o_test.to(self.device), N_p=N_p, dyn=dyn, prop=prop, re=re)
            s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
            mse_pf = ((s_est - s_test.to(self.device))**2).detach().cpu().numpy()
            print(f'RSPF test mse = {mse_pf.mean():.8f}')
            mse_pfdf = pd.DataFrame(np.squeeze(mse_pf))

            if not os.path.isdir('results'): 
                os.mkdir('results')
            dir = os.path.join('results', 'mse')
            if not os.path.isdir(dir): 
                os.mkdir(dir)
            mse_pfdf.to_csv(f'./{dir}/rspf_{dyn}_{prop}_{re}')
 

class DPF(nn.Module): 
    def __init__(self, nnm, mu_u, mu_v, args, learning_rate, device):
        super().__init__()
        self.mat_P, _, _, _, _, self.beta = args
        self.hidden = 8
        if nnm: 
            self.dynamic = dynamic_NN(1, self.hidden, 1)
            self.measure = dynamic_NN(1, self.hidden, 1)
        else: 
            self.co_A = nn.Parameter(torch.Tensor(1).uniform_(-1, 1))
            self.co_B = nn.Parameter(torch.Tensor(1).uniform_(-4, 4))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        
        self.mu_u = mu_u
        self.sigma_u = nn.Parameter(torch.Tensor(1).uniform_(0.1, 0.5))
        self.mu_v = mu_v
        self.sigma_v = nn.Parameter(torch.Tensor(1).uniform_(0.1, 0.5))

        self.nnm = nnm
        self.lr = learning_rate
        self.device = device
        self.loss = nn.MSELoss()
        self.optim = torch.optim.SGD(self.parameters(), lr = learning_rate, momentum=0.9)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10, 20, 30, 40, 50], gamma=0.5)

    def initial(self, size=(1, 1)): 
        return torch.rand(size=size)-0.5

    def state(self, s_p): 
        s_t = self.co_A * s_p + self.co_B + torch.randn(s_p.shape, device=self.device) * self.sigma_u.data + self.mu_u
        return torch.where(s_t==0., s_t+1e-4, s_t)

    def state_nn(self, s_p): 
        return self.dynamic(s_p).view(s_p.shape) + torch.randn(s_p.shape, device=self.device) * self.sigma_u.data + self.mu_u

    def measure_nn(self, s_t): 
        return self.measure(s_t).view(s_t.shape) + torch.randn(s_t.shape, device=self.device) * self.sigma_v.data + self.mu_v

    def filtering(self, o, N_p, re): 
        batch, _, T = o.shape
        s_list = torch.zeros(batch, N_p, T, device=self.device)
        w_list = torch.zeros(batch, N_p, T, device=self.device)
        s_list[:, :, 0] = self.initial(size=(batch, N_p))
        s_list_re = s_list.clone().detach()
        if self.nnm: 
            o_list = s_list.clone().detach()
        w_list[:, :, 0] = 1/N_p
        w_list_re = w_list.clone().detach()
        w_likelihood = torch.zeros(batch, T, device=self.device)
        w_likelihood[:, 0] = torch.log(w_list[:, :, 0].mean(dim=1))
        for t in range(1, T): 
            if self.nnm: 
                s_list[:, :, [t]] = self.state_nn(s_list_re[:, :, [t-1]])
                o_list[:, :, [t]] = self.measure_nn(s_list[:, :, [t]])          
                w_list[:, :, [t]], w_likelihood[:, [t]] = weights_bootstrap_nn(self, w_list_re[:, :, [t-1]], o_list[:, :, [t]], o[:, :, [t]])
            else: 
                s_list[:, :, [t]] = self.state(s_list_re[:, :, [t-1]])  
                w_list[:, :, [t]], w_likelihood[:, [t]] = weights_bootstrap_sin(self, w_list_re[:, :, [t-1]], s_list[:, :, [t]], o[:, :, [t]])

            w_list_re[:, :, t] = w_list[:, :, t].clone().detach()
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < (s_list.shape[1] + 1)

            if ESS.sum(): 
                re_index = torch.where(ESS)[0]
                index = torch.arange(N_p, dtype=torch.long, device=self.device).tile(batch, 1)                
                if re=="sys": 
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t], device=self.device)
                else: 
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t], device=self.device)
                w_list_re[re_index, :, t] = 1 / N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                index_flatten = index.view(-1)
                s_trans = s_list[batch_index, index_flatten, :t+1].clone().detach()
                s_list_re[:, :, :t+1] = s_trans.view(len(re_index), N_p, -1)

        return s_list, w_list, w_likelihood

    def forward(self, o_data, N_p, re):  
        s_parlist, w_parlist, w_likelihood = self.filtering(o_data, N_p=N_p, re=re)
        s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
        return s_est, w_likelihood
    
    def train(self, train_data, val_data, N_iter, N_p, dyn, re): 
        N_step = len(train_data)
        l = np.ones(N_iter) * 1e2
        if not os.path.isdir('results'): 
            os.mkdir('results')
        dir = os.path.join('results', 'bestval')
        if not os.path.isdir(dir): 
            os.mkdir(dir)
        for epoch in range(N_iter): 
            for i, (_, s_train, o_train) in enumerate(train_data): 
                s_est, w_likelihood = self(o_train.to(self.device), N_p, re)
                # loss_sample = ((s_est - s_data)**2).mean(dim=2, keepdim=True)
                # nll = - torch.distributions.Normal(loc=s_est, scale=self.sigma_u).log_prob(s_train)
                # nll_2 = - w_likelihood.sum(dim=1)
                loss = self.loss(s_est, s_train.to(self.device))
                # loss_sample.mean(dim=0, keepdim=True)
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()
                print(f'epoch{epoch+1}/{N_iter}, step{i+1}/{N_step}: training loss = {loss.item():.8f}')
            self.optim_scheduler.step()
            with torch.no_grad(): 
                for _, s_val, o_val in val_data: 
                    s_est, _ = self(o_val.to(self.device), N_p, re)
                    # loss = ((s_est - s_val)**2).mean()
                    loss = self.loss(s_est, s_val.to(self.device))
                    if loss.item() < l.min(): 
                        torch.save(self, f'./{dir}/dpf_nn{self.nnm}_{dyn}_{re}_lr{self.lr}_hidden{self.hidden}')
                    l[epoch] = loss
            print(f'epoch{epoch+1}/{N_iter}: validation loss = {loss:.8f}')
            
        print(l)

    def test(self, test_data, N_p, dyn, re): 
        best_val = torch.load(f'./results/bestval/dpf_nn{self.nnm}_{dyn}_{re}_lr{self.lr}_hidden{self.hidden}')
        best_val.device = self.device
        with torch.no_grad(): 
            for _, s_test, o_test in test_data: 
                s_est, _ = best_val(o_test.to(self.device), N_p, re)
                loss = self.loss(s_est, s_test.to(self.device))
                print(f'DPF test loss = {loss.item():.8f}')
            mse_dpf = ((s_est - s_test.to(self.device))**2).detach().cpu().numpy()
            print(f'DPF test mse = {mse_dpf.mean():.8f}')
            mse_dpfdf = pd.DataFrame(np.squeeze(mse_dpf))
            
            dir = os.path.join('results', 'mse')
            if not os.path.isdir(dir): 
                os.mkdir(dir)
            mse_dpfdf.to_csv(f'./{dir}/dpf_nn{self.nnm}_{dyn}_{re}_lr{self.lr}_hidden{self.hidden}')
        

class RSDPF(nn.Module, RSPF): 
    def __init__(self, N_m, nnm, mu_u, mu_v, args, learning_rate, device): 
        super().__init__()
        self.mat_P, _, _, _, _, self.beta = args
        self.N_m = N_m
        self.hidden = 8
        if nnm: 
            self.dynamic = dynamic_RSNN(1, self.hidden, 1)
            self.measure = dynamic_RSNN(1, self.hidden, 1)
        else: 
            self.co_A = nn.Parameter(torch.Tensor(self.N_m).uniform_(-1, 1))
            self.co_B = nn.Parameter(torch.Tensor(self.N_m).uniform_(-4, 4))
            self.co_C = nn.Parameter(self.co_A.data.clone().detach())
            self.co_D = nn.Parameter(self.co_B.data.clone().detach())
        
        self.mu_u = mu_u
        self.sigma_u = nn.Parameter(torch.Tensor(1).uniform_(0.1, 0.5))
        self.mu_v = mu_v
        self.sigma_v = nn.Parameter(torch.Tensor(1).uniform_(0.1, 0.5))

        self.nnm = nnm
        self.lr = learning_rate
        self.device = device
        self.loss = nn.MSELoss()
        self.optim = torch.optim.SGD(self.parameters(), lr = learning_rate, momentum=0.9)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[10, 20, 30, 40, 50], gamma=0.5)

    def state(self, m_t, s_p): 
        s_t = self.co_A[m_t] * s_p + self.co_B[m_t] + torch.randn(s_p.shape, device=self.device) * self.sigma_u[0] + self.mu_u
        return torch.where(s_t==0., s_t+1e-4, s_t)

    def state_nn(self, m_t, s_p): 
        s_t = torch.empty(s_p.shape, device=self.device)
        s_t[m_t==0] = self.dynamic(s_p[m_t==0], 0)
        s_t[m_t==1] = self.dynamic(s_p[m_t==1], 1)
        s_t[m_t==2] = self.dynamic(s_p[m_t==2], 2)
        s_t[m_t==3] = self.dynamic(s_p[m_t==3], 3)
        s_t[m_t==4] = self.dynamic(s_p[m_t==4], 4)
        s_t[m_t==5] = self.dynamic(s_p[m_t==5], 5)
        s_t[m_t==6] = self.dynamic(s_p[m_t==6], 6)
        s_t[m_t==7] = self.dynamic(s_p[m_t==7], 7)
        return s_t + torch.randn(s_p.shape, device=self.device) * self.sigma_u[0] + self.mu_u

    def measure_nn(self, m_t, s_t): 
        o_t = torch.empty(s_t.shape, device=self.device)
        o_t[m_t==0] = self.measure(s_t[m_t==0], 0)
        o_t[m_t==1] = self.measure(s_t[m_t==1], 1)
        o_t[m_t==2] = self.measure(s_t[m_t==2], 2)
        o_t[m_t==3] = self.measure(s_t[m_t==3], 3)
        o_t[m_t==4] = self.measure(s_t[m_t==4], 4)
        o_t[m_t==5] = self.measure(s_t[m_t==5], 5)
        o_t[m_t==6] = self.measure(s_t[m_t==6], 6)
        o_t[m_t==7] = self.measure(s_t[m_t==7], 7)
        return o_t + torch.randn(s_t.shape, device=self.device) * self.sigma_v[0] + self.mu_v

    def filtering(self, o, N_p, dyn, prop, re): 
        batch, _, T = o.shape
        m_list = torch.zeros(batch, N_p, T, dtype=torch.long, device=self.device)
        s_list = torch.zeros(batch, N_p, T, device=self.device)
        w_list = torch.zeros(batch, N_p, T, device=self.device)
        m_list[:, :, 0], s_list[:, :, 0] = self.initial(size=(batch, N_p))
        m_list_re = m_list.clone().detach()
        s_list_re = s_list.clone().detach()
        if self.nnm: 
            o_list = s_list.clone().detach()
        w_list[:, :, 0] = 1/N_p
        w_list_re = w_list.clone().detach()
        w_likelihood = torch.zeros(batch, T, device=self.device)
        w_likelihood[:, 0] = torch.log(w_list[:, :, 0].mean(dim=1))
        for t in range(1, T): 
            if prop=="Boot": 
                if dyn=="Poly": 
                    m_list[:, :, t] = self.Polyaurn_dynamic(m_list_re[:, :, :t])
                elif dyn=="Mark": 
                    m_list[:, :, t] = self.Markov_dynamic(m_list_re[:, :, t-1])
            elif prop=="Uni": 
                m_list[:, :, t] = self.propuni_dynamic(size=(batch, N_p))
            elif prop=='Deter': 
                m_list[:, :, t] = torch.arange(len(self.beta)).tile(batch, int(N_p/len(self.beta)))
            
            m_list_re[:, :, t] = m_list[:, :, t].clone().detach()

            if self.nnm: 
                s_list[:, :, [t]] = self.state_nn(m_list[:, :, [t]], s_list_re[:, :, [t-1]])
                o_list[:, :, [t]] = self.measure_nn(m_list[:, :, [t]], s_list[:, :, [t]])          
                if prop=="Boot": 
                    w_list[:, :, [t]], w_likelihood[:, [t]] = weights_bootstrap_nn(self, w_list_re[:, :, [t-1]], o_list[:, :, [t]], o[:, :, [t]])
                else: 
                    if dyn=="Mark": 
                        w_list[:, :, [t]] = weights_proposal_nn(self, w_list_re[:, :, [t-1]], m_list_re[:, :, t-1:t+1], o_list[:, :, [t]], o[:, :, [t]], dyn)
                    else: 
                        w_list[:, :, [t]] = weights_proposal_nn(self, w_list_re[:, :, [t-1]],  m_list_re[:, :, :t+1], o_list[:, :, [t]], o[:, :, [t]], dyn)
            else: 
                s_list[:, :, [t]] = self.state(m_list[:, :, [t]], s_list_re[:, :, [t-1]])  
                if prop=="Boot": 
                    w_list[:, :, [t]], w_likelihood[:, [t]] = weights_bootstrap(self, w_list_re[:, :, [t-1]], m_list[:, :, [t]], s_list[:, :, [t]], o[:, :, [t]])
                else:
                    if dyn=="Mark": 
                        w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, t-1:t+1], s_list[:, :, [t]], o[:, :, [t]], dyn)
                    else: 
                        w_list[:, :, [t]] = weights_proposal(self, w_list_re[:, :, [t-1]], m_list_re[:, :, :t+1], s_list[:, :, [t]], o[:, :, [t]], dyn)

            w_list_re[:, :, t] = w_list[:, :, t].clone().detach()
            ESS = 1 / (w_list[:, :, t]**2).sum(dim=1) < (s_list.shape[1] + 1)
            if ESS.sum(): 
                re_index = torch.where(ESS)[0]
                index = torch.arange(N_p, dtype=torch.long, device=self.device).tile(batch, 1)                
                if re=="sys": 
                    index[re_index, :] = resample_systematic(w_list[re_index, :, t], device=self.device)
                else: 
                    index[re_index, :] = resample_multinomial(w_list[re_index, :, t], device=self.device)
                w_list_re[re_index, :, t] = 1 / N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                index_flatten = index.view(-1)
                m_trans = m_list[batch_index, index_flatten, :t+1].clone().detach()
                m_list_re[:, :, :t+1] = m_trans.view(len(re_index), N_p, -1)
                s_trans = s_list[batch_index, index_flatten, :t+1].clone().detach()
                s_list_re[:, :, :t+1] = s_trans.view(len(re_index), N_p, -1)

        return m_list, s_list, w_list, w_likelihood

    def forward(self, o_data, N_p, dyn, prop, re):  
        m_parlist, s_parlist, w_parlist, w_likelihood = self.filtering(o_data, N_p=N_p, dyn=dyn, prop=prop, re=re)
        s_est = (w_parlist * s_parlist).sum(dim=1, keepdim=True)
        return s_est, w_likelihood
    
    def train(self, train_data, val_data, N_iter=50, N_p=200, dyn="Mark", prop="Boot", re="mul"): 
        N_step = len(train_data)
        l = np.ones(N_iter) * 1e2
        if not os.path.isdir('results'): 
            os.mkdir('results')
        dir = os.path.join('results', 'bestval')
        if not os.path.isdir(dir): 
            os.mkdir(dir)
        for epoch in range(N_iter): 
            for i, (_, s_train, o_train) in enumerate(train_data): 
                s_est, w_likelihood = self(o_train.to(self.device), N_p, dyn, prop, re)
                # loss_sample = ((s_est - s_data)**2).mean(dim=2, keepdim=True)
                # nll = - torch.distributions.Normal(loc=s_est, scale=self.sigma_u).log_prob(s_train)
                # nll_2 = - w_likelihood.sum(dim=1)
                loss = self.loss(s_est, s_train.to(self.device))
                # loss_sample.mean(dim=0, keepdim=True)
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()
                print(f'epoch{epoch+1}/{N_iter}, step{i+1}/{N_step}: training loss = {loss.item():.8f}')
            self.optim_scheduler.step()
            with torch.no_grad(): 
                for _, s_val, o_val in val_data: 
                    s_est, _ = self(o_val.to(self.device), N_p, dyn, prop, re)
                    # loss = ((s_est - s_val)**2).mean()
                    loss = self.loss(s_est, s_val.to(self.device))
                    if loss.item() < l.min(): 
                        torch.save(self, f'./{dir}/rsdpf_nn{self.nnm}_{dyn}_{prop}_{re}_lr{self.lr}_hidden{self.hidden}')
                    l[epoch] = loss
            print(f'epoch{epoch+1}/{N_iter}: validation loss = {loss:.8f}')
            
        print(l)

    def test(self, test_data, N_p=2000, dyn="Mark", prop="Boot", re="mul"): 
        best_val = torch.load(f'./results/bestval/rsdpf_nn{self.nnm}_{dyn}_{prop}_{re}_lr{self.lr}_hidden{self.hidden}')
        best_val.device = self.device
        with torch.no_grad(): 
            for _, s_test, o_test in test_data: 
                s_est, _ = best_val(o_test.to(self.device), N_p, dyn, prop, re)
                loss = self.loss(s_est, s_test.to(self.device))
                print(f'RSDPF test loss = {loss.item():.8f}')
            mse_dpf = ((s_est - s_test.to(self.device))**2).detach().cpu().numpy()
            print(f'RSDPF test mse = {mse_dpf.mean():.8f}')
            mse_dpfdf = pd.DataFrame(np.squeeze(mse_dpf))

            dir = os.path.join('results', 'mse')
            if not os.path.isdir(dir): 
                os.mkdir(dir)
            mse_dpfdf.to_csv(f'./{dir}/rsdpf_nn{self.nnm}_{dyn}_{prop}_{re}_lr{self.lr}_hidden{self.hidden}')


class MMPF(RSPF): 

    def __init__(self, N_m, args, mu_u, sigma_u, mu_v, sigma_v, gamma, device): 
        self.N_m = N_m
        _, A, B, C, D, _ = args
        self.co_A = A.view(1, -1, 1, 1)
        self.co_B = B.view(1, -1, 1, 1)
        self.co_C = C.view(1, -1, 1, 1)
        self.co_D = D.view(1, -1, 1, 1)
        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.mu_v = mu_v
        self.sigma_v = sigma_v
        self.gamma = gamma
        self.device = device

    def state(self, s_p): 
        return self.co_A * s_p + self.co_B + torch.randn(s_p.shape, device=self.device) * self.sigma_u + self.mu_u

    def filtering(self, o, N_p, re): 
        batch, _, T = o.shape
        N_pm = int(N_p/self.N_m)
        s_list = torch.zeros(batch, self.N_m, N_pm, T, device=self.device)
        w_list = torch.zeros(batch, self.N_m, N_pm, T, device=self.device)   
        pi_list = torch.zeros(batch, self.N_m, 1, T, device=self.device)     
        _, s_list[:, :, :, 0] = self.initial(size=(batch, self.N_m, N_pm))
        # s_list[:, :, :, 0] = s
        s_list_re = s_list.clone().detach()
        w_list[:, :, :, 0] = self.N_m/N_p
        w_list_re = w_list.clone().detach()
        pi_list[:, :, :, 0] = 1/self.co_A.shape[1]
        for t in range(1, T): 
            s_list[:, :, :, [t]] = self.state(s_list_re[:, :, :, [t-1]])  
            pi_list[:, :, :, [t]] = pi_list[:, :, :, [t-1]]**self.gamma / (pi_list[:, :, :, [t-1]]**self.gamma).sum(dim=1, keepdim=True)
            w_list[:, :, :, [t]], pi_list[:, :, :, [t]] = weights_mmpf(self, w_list_re[:, :, :, [t-1]], s_list[:, :, :, [t]], pi_list[:, :, :, [t]], o[:, :, [t]])
            w_list_re[:, :, :, t] = w_list[:, :, :, t].clone().detach()
           
            ESS = 1 / (w_list[:, :, :, t]**2).sum(dim=2) < (s_list.shape[2] + 1)
            if ESS.sum(): 
                re_index1, re_index2 = torch.where(ESS)[0], torch.where(ESS)[1]
                index = torch.arange(N_pm, dtype=torch.long, device=self.device).tile(batch, self.N_m, 1)                
                if re=="sys":                    
                    index[re_index1, re_index2, :] = resample_systematic_mmpf(w_list[re_index1, re_index2, :, t], len(torch.unique(re_index1)), device=self.device)
                else: 
                    index[re_index1, re_index2, :] = resample_multinomial_mmpf(w_list[re_index1, re_index2, :, t], len(torch.unique(re_index1)), device=self.device)
                w_list_re[re_index1, re_index2, :, t] = self.N_m/N_p
                
                batch_index = torch.arange(batch, dtype=torch.long).tile(N_p, 1).T.reshape(-1)
                m_index = torch.arange(self.N_m, dtype=torch.long).tile(N_pm, 1).T.reshape(-1).tile(batch)
                index_flatten = index.view(-1)
                s_trans = s_list[batch_index, m_index, index_flatten, :t+1].clone().detach()
                
                s_list_re[:, :, :, :t+1] = s_trans.view(batch, self.N_m, N_pm, -1)

        return pi_list, s_list, w_list

    def test(self, test_data, N_p=2000, dyn="Mark", re="mul"): 
        for _, s_test, o_test in test_data: 
            pi_list, s_parlist, w_parlist = self.filtering(o_test.to(self.device), N_p=N_p, re=re)
            s_est = (pi_list * (w_parlist * s_parlist).sum(dim=2, keepdim=True)).sum(dim=1)
            mse_mmpf = ((s_est - s_test.to(self.device))**2).detach().cpu().numpy()
            print(f'MMPF test mse = {mse_mmpf.mean():.8f}')
            mse_mmpfdf = pd.DataFrame(np.squeeze(mse_mmpf))

            if not os.path.isdir('results'): 
                os.mkdir('results')
            dir = os.path.join('results', 'mse')
            if not os.path.isdir(dir): 
                os.mkdir(dir)
            mse_mmpfdf.to_csv(f'./{dir}/mmpf_{dyn}_gamma{self.gamma:.1f}_{re}')



