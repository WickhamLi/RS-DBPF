import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

class RSPF: 
    def __init__(self, tran_matrix, A, B, C, D, mu_u=0., sigma_u=np.sqrt(0.1), mu_v=0., sigma_v=np.sqrt(0.1), beta=np.array(1)): 
        self.mat_P = tran_matrix
        self.co_A = A
        self.co_B = B
        self.co_C = C
        self.co_D = D
        self.mu_u = mu_u
        self.sigma_u = sigma_u
        self.mu_v = mu_v
        self.sigma_v = sigma_v
        self.beta = beta.reshape(1, -1)
        
    def initial(self, size=1): 
        return np.random.choice([i for i in range(int(8))], replace=True, size=size), np.random.uniform(-0.5, 0.5, size=size)
        
    def Markov_dynamic(self, m_p, size=1): 
#         data size = m_p.size
#         model number = cum.shape[-1]
        sample = np.random.rand(size)
        m_t = np.empty(size)
#         for i in range(size): 
#             k = 0
#             j = -1
#             while sample[i] > k: 
#                 j += 1
#                 k += self.mat_P[m_p[i]][j]
#             m_t[i] = j
#         m_t = m_t.astype(int)
        cum = np.cumsum(self.mat_P[m_p], axis=1)
        boolean = np.empty((size, cum.shape[-1]))
        for i in range(size): 
            boolean[i, :] = sample[i] < cum[i, :]
        for j in range(size): 
            for k in range(cum.shape[-1]): 
                if boolean[j][k]: 
                    m_t[j] = k
                    break
        m_t = m_t.astype(int)
        return m_t
    
    def Polyaurn_dynamic(self, m_p, size=1): 
#         time step = m_p.shape[0]
#         data size = m_p.shape[-1]
#         model number = self.beta.size
        m_p = np.array(m_p)
        sample = np.random.rand(size)
        m_t = np.empty(size)
        alpha = np.zeros((size, self.beta.size))
        for i in range(size): 
            unique, counts = np.unique(m_p[:, i], return_counts=True)
            alpha[i, unique] = counts
        post = (alpha + self.beta)/(alpha + self.beta).sum(axis=1, keepdims=True)
#         for i in range(size): 
#             k = 0
#             j = -1
#             while sample[i] > k: 
#                 j += 1
#                 k += post[i, j]
#             m_t[i] = j
#         m_t = m_t.astype(int)
        cum = np.cumsum(post, axis=1)
        boolean = np.empty((size, self.beta.size))
        for i in range(size): 
            boolean[i, :] = sample[i] < cum[i, :]
        for j in range(size): 
            for k in range(cum.shape[-1]): 
                if boolean[j][k]: 
                    m_t[j] = k
                    break
        m_t = m_t.astype(int)
        return m_t
    
    def propuni_dynamic(self, size=1): 
#         data size = m_p.size
#         model number = cum.shape[-1]
        m_t = np.random.choice(len(self.co_A), size=size, replace=True)
        m_t = m_t.astype(int)
        return m_t
        
    def state(self, m_t, s_p, size=1): 
        return self.co_A[m_t] * s_p.reshape(-1, 1) + self.co_B[m_t] + np.random.normal(loc=self.mu_u, scale=self.sigma_u, size=size).reshape(-1, 1)
        
    def obs(self, m_t, s_t, size=1): 
        return self.co_C[m_t] * np.sqrt(np.abs(s_t)).reshape(-1, 1) + self.co_D[m_t] + np.random.normal(loc=self.mu_v, scale=self.sigma_v, size=size).reshape(-1, 1)
        

def create_parameters(N_m=8): 
    p = np.array([0.8]*N_m)
    P = np.diag(p)
    for i in range(N_m): 
        for j in range(N_m): 
            if i == j-1 or i-N_m+1 == j: 
                P[i][j] = 0.15
            elif i != j: 
                P[i][j] = 1/120
#     P = np.eye(8)
    A = np.array([[-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9]]).T
    B = np.array([[0., -2., 2., -4., 0., 2., -2., 4.]]).T
    C = A.copy()
    D = B.copy()
    beta = np.array([1]*N_m)
    return P, A, B, C, D, beta

def generate_data(T, model, size=1, dyn=0): 
    m = []
    s = []
    o = []
    m_0, s_0 = model.initial(size=size)
    o_0 = model.obs(m_0, s_0)
    m.append(m_0.reshape(-1))
    s.append(s_0.reshape(-1))
    o.append(o_0.reshape(-1))
    for i in range(1, T):
        if dyn: 
            m.append(model.Polyaurn_dynamic(m))
        else: 
            m.append(model.Markov_dynamic(m[i-1]))
        s.append(model.state(m[i], s[i-1]))
        o.append(model.obs(m[i], s[i]))
    return m, s, o

def weights_bootstrap(model, w_list, m_list, s_list, o): 
    lw = np.log(w_list)
    for i in range(len(s_list)): 
        lw[i] += stats.norm(loc=model.co_C[m_list[i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[i]], scale=np.sqrt(0.1)).logpdf(o)
#         w += 10**(-300)
    w = np.exp(lw - lw.max())
    return w/w.sum()

def weights_proposal(model, w_list, m_list, s_list, o, switch): 
    lw = np.log(w_list)
    if m_list.shape[0]>1: 
        if switch==2: 
            lp = np.log(model.mat_P[m_list[-2, :], m_list[-1, :]])
    else:
        lp = np.log([1/len(model.co_A)]*len(w_list))
    for i in range(len(s_list)): 
        lw[i] += stats.norm(loc=model.co_C[m_list[-1, i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[-1, i]], scale=np.sqrt(0.1)).logpdf(o) + lp[i] - np.log((1/len(model.co_A)))
#         w += 10**(-300)
#         print(lw[i])
    w = np.exp(lw - lw.max())
    return w/w.sum()


def inv_cdf(cum, ws): 
    index = np.empty(len(ws), dtype=int)
    w = ws[0]
    k = 0
    for i in range(len(ws)): 
        while cum[i] > w: 
            k += 1
            w += ws[k]
        index[i] = k
    return index
    

def resample_systematic(N_p, ws): 
    cum = (np.random.rand(1) + np.arange(N_p)) / N_p
    return inv_cdf(cum, ws)
    
def resample_multinomial(N_p, ws): 
    uni = np.cumsum(-np.log(np.random.rand(N_p+1)))
    cum = uni[:-1] / uni[-1]
    return inv_cdf(cum, ws)

def filtering(T, model, m, s, o, N_p=200, dyn=2, re=0, switch=2): 
    s_list = np.zeros((T, N_p))
    m_list = np.zeros((T, N_p), dtype=int)
    w_list = np.zeros((T, N_p))
    # m_list[0, :], s_list[0, :] = model.initial(size=N_p)
    m_list[0, :] = m
    s_list[0, :] = s
    w_list[0, :] = 1/N_p
    for t in range(1, T): 
        if dyn==1: 
            m_list[t, :] = model.Polyaurn_dynamic(m_list[:t, :], size=N_p)
        elif dyn==2: 
            m_list[t, :] = model.Markov_dynamic(m_list[t-1, :], size=N_p)
        elif dyn==0: 
            m_list[t, :] = model.propuni_dynamic(size=N_p)
        else: 
            m_list[t, ] = np.array([i for i in range(len(model.co_A))]*int(N_p/len(model.co_A)))
        s_list[t, :] = np.squeeze(model.state(m_list[t, :], s_list[t-1, :], size=N_p))
        if dyn==1 or dyn==2: 
            w_list[t, :] = weights_bootstrap(model, w_list[t-1, :], m_list[t, :], s_list[t, :], o[t].astype(float))
        else:
            w_list[t, :] = weights_proposal(model, w_list[t-1, :], m_list[:t, :], s_list[t, :], o[t].astype(float), switch)
        ESS = 1 / np.sum(w_list[t, :]**2)
#         print(ESS)
        if ESS < s_list.shape[-1]: 
            if re: 
                index = resample_systematic(len(w_list[t, :]), w_list[t, :])
            else: 
                index = resample_multinomial(len(w_list[t, :]), w_list[t, :])
            index = index.astype(int)
            w_list[t, :] = 1 / len(w_list[t, :])
            m_list[t, :] = m_list[t, index]
            s_list[t, :] = s_list[t, index]
    return m_list, s_list, w_list

def MSE(w_list, s_list, s): 
    s = np.squeeze(s)
    s_est = np.array([np.dot(w_list[t, :], s_list[t, :]) for t in range(s_list.shape[0])])
    mse = (s_est - s)**2
    mse_cum = np.cumsum(mse)
    return mse, mse_cum
