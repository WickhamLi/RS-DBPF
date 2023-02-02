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
        self.beta = beta
        
    def initial(self, size=1): 
        return np.random.choice([i for i in range(int(8))], replace=True, size=size), np.random.uniform(-0.5, 0.5, size=size)
        
    def Markov_dynamic(self, m_p): 
        batch, N_p = m_p.shape
        N_m = self.mat_P.shape[-1]

        sample = np.random.rand(batch, N_p, 1)
        m_t = np.ones((batch, N_p, 1), dtype=int) * N_m
#         for i in range(size): 
#             k = 0
#             j = -1
#             while sample[i] > k: 
#                 j += 1
#                 k += self.mat_P[m_p[i]][j]
#             m_t[i] = j
#         m_t = m_t.astype(int)
        cum = np.cumsum(self.mat_P[m_p, :], axis=2)
        m_t -= np.sum(sample < cum, axis=2, keepdims=True)

        return m_t.reshape(batch, N_p)
    
    def Polyaurn_dynamic(self, m_p): 
        batch, N_p = m_p.shape[:2]
        N_m = self.beta.size

        sample = np.random.rand(batch, N_p, 1)
        m_t = np.ones((batch, N_p, 1), dtype=int) * N_m
        alpha = np.zeros((batch, N_p, N_m))

        for m_index in range(N_m): 
            alpha[:, :, m_index] = np.sum(m_p == m_index, axis=2)
        
        beta = self.beta[np.newaxis, np.newaxis, :]
        post = (alpha + beta)/(alpha + beta).sum(axis=2, keepdims=True)
#         for i in range(size): 
#             k = 0
#             j = -1
#             while sample[i] > k: 
#                 j += 1
#                 k += post[i, j]
#             m_t[i] = j
#         m_t = m_t.astype(int)
        cum = np.cumsum(post, axis=2)
        m_t -= np.sum(sample < cum, axis=2, keepdims=True)

        return m_t.reshape(batch, N_p)
    
    def propuni_dynamic(self, size=1): 
        m_t = np.random.choice(len(self.co_A), size=size, replace=True)
        m_t = m_t.astype(int)
        return m_t
        
    def state(self, m_t, s_p): 
        return self.co_A[m_t] * s_p + self.co_B[m_t] + np.random.normal(loc=self.mu_u, scale=self.sigma_u, size=s_p.shape)
        
    def obs(self, m_t, s_t): 
        return self.co_C[m_t] * np.sqrt(np.abs(s_t)) + self.co_D[m_t] + np.random.normal(loc=self.mu_v, scale=self.sigma_v, size=s_t.shape)
        

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
    A = np.array([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9])
    B = np.array([0., -2., 2., -4., 0., 2., -2., 4.])
    C = A.copy()
    D = B.copy()
    beta = np.array([1]*N_m)
    return P, A, B, C, D, beta

def generate_data(T, model, batch=1, dyn="Mark"): 
    m = np.zeros((batch, 1, T), dtype=int)
    s = np.zeros((batch, 1, T))
    o = np.zeros((batch, 1, T))
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
    lw = np.log(w_list)
    o = np.ones(s_list.shape) * o
    o -= (model.co_C[m_list] * np.sqrt(np.abs(s_list)) + model.co_D[m_list])
    lw += stats.norm(scale=np.sqrt(0.1)).logpdf(o)
#     for i in range(len(s_list)): 
#         lw[i] += stats.norm(loc=model.co_C[m_list[i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[i]], scale=np.sqrt(0.1)).logpdf(o)
# #         w += 10**(-300)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    return w/w.sum(axis=1, keepdims=True)

def weights_proposal(model, w_list, m_list, s_list, o, dyn): 
    lw = np.log(w_list)
    o = np.ones(s_list.shape) * o
    o -= (model.co_C[m_list[:, :, -1]] * np.sqrt(np.abs(s_list)) + model.co_D[m_list[:, :, -1]])
    if dyn=="Mark": 
        lp = np.log(model.mat_P[m_list[:, :, -2], m_list[:, :, -1]])
    else: 
        batch = m_list.shape[0]
        N_p = m_list.shape[1]
        alpha = np.zeros((batch, N_p, len(model.beta)))
        for m_index in range(len(model.beta)): 
            alpha[:, :, m_index] = np.sum(m_list[:, :, :-1] == m_index, axis=2)
        beta = model.beta[np.newaxis, np.newaxis, :]
        prob = (alpha + beta)/(alpha + beta).sum(axis=2, keepdims=True)
        batch_index = tuple(i//N_p for i in range(batch * N_p))
        pars_index = tuple([i for i in range(N_p)] * batch)
        index_flatten = tuple(m_list[:, :, -1].reshape(-1))
        lp = np.log(prob[batch_index, pars_index, index_flatten].reshape(batch, N_p))

    lw += stats.norm(scale=np.sqrt(0.1)).logpdf(o) + lp - np.log((1/len(model.co_A)))
    # for i in range(len(s_list)): 
    #     lw[i] += stats.norm(loc=model.co_C[m_list[-1, i]] * np.sqrt(np.abs(s_list[i])) + model.co_D[m_list[-1, i]], scale=np.sqrt(0.1)).logpdf(o) + lp[i] - np.log((1/len(model.co_A)))
#         w += 10**(-300)
#         print(lw[i])
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    return w/w.sum(axis=1, keepdims=True)


def inv_cdf(cum, ws): 
    index = np.ones(ws.shape, dtype=int) * cum.shape[-1]
    w_cum = np.cumsum(ws, axis=1)
    for i in range(cum.shape[-1]): 
        index[:, [i]] -= np.sum(cum[:, [i]] < w_cum, axis=1, keepdims=True)
        # while cum[i] > w: 
        #     k += 1
        #     w += ws[k]
        # index[i] = k
    return index
    

def resample_systematic(ws): 
    batch, N_p = ws.shape
    cum = (np.random.rand(batch, 1) + np.tile(np.arange(N_p), batch).reshape(batch, -1)) / np.array([N_p] * batch).reshape(-1, 1)
    return inv_cdf(cum, ws)
    
def resample_multinomial(ws): 
    batch, N_p = ws.shape
    uni = np.cumsum(-np.log(np.random.rand(batch, N_p+1)), axis=1)
    cum = uni[:, :-1] / uni[:, [-1]]
    return inv_cdf(cum, ws)

def filtering(model, o, N_p=200, dyn="Mark", prop="Boot", re="mul"): 
    batch, T = o.shape
    m_list = np.zeros((batch, N_p, T), dtype=int)
    s_list = np.zeros((batch, N_p, T))
    w_list = np.zeros((batch, N_p, T))
    m_list[:, :, 0], s_list[:, :, 0] = model.initial(size=(batch, N_p))
    # m_list[0, :] = m
    # s_list[0, :] = s
    w_list[:, :, 0] = 1/N_p
    for t in range(1, T): 
        if prop=="Boot": 
            if dyn=="Poly": 
                m_list[:, :, t] = model.Polyaurn_dynamic(m_list[:, :, :t])
            elif dyn=="Mark": 
                m_list[:, :, t] = model.Markov_dynamic(m_list[:, :, t-1])
        elif prop=="Uni": 
            m_list[:, :, t] = model.propuni_dynamic(size=(batch, N_p))
        elif prop=='Deter': 
            m_list[:, :, t] = np.array([i for i in range(len(model.co_A))]*int(batch * N_p/len(model.co_A))).reshape(batch, N_p)
        s_list[:, :, t] = model.state(m_list[:, :, t], s_list[:, :, t-1])
        if prop=="Boot": 
            w_list[:, :, t] = weights_bootstrap(model, w_list[:, :, t-1], m_list[:, :, t], s_list[:, :, t], o[:, [t]])
        else:
            if dyn=="Mark": 
                w_list[:, :, t] = weights_proposal(model, w_list[:, :, t-1], m_list[:, :, t-1:t+1], s_list[:, :, t], o[:, [t]], dyn)
            else: 
                w_list[:, :, t] = weights_proposal(model, w_list[:, :, t-1], m_list[:, :, :t+1], s_list[:, :, t], o[:, [t]], dyn)
        ESS = 1 / np.sum(w_list[:, :, t]**2, axis=1) < s_list.shape[1]
#         print(ESS)
        if ESS.sum(): 
            index = np.tile(np.arange(N_p), batch).reshape(batch, -1)
            if re=="sys": 
                re_index = np.array(np.where(ESS)).reshape(-1)
                index[re_index, :] = resample_systematic(w_list[re_index, :, t])
            else: 
                re_index = np.array(np.where(ESS)).reshape(-1)
                index[re_index, :] = resample_multinomial(w_list[re_index, :, t])
            w_list[:, :, t] = 1 / N_p
            # for b in range(batch): 
            #     for n in range(N_p): 
            #         m_list[b, n, t] = np.copy(m_list[b, index[b, n], t])
            #         s_list[b, n, t] = np.copy(m_list[b, index[b, n], t])
            
            batch_index = tuple(i//N_p for i in range(batch * N_p))
            index_flatten = tuple(index.reshape(-1))
            m_trans = m_list[batch_index, index_flatten, [t]]
            m_list[:, :, t] = m_trans.reshape(batch, -1)
            s_trans = s_list[batch_index, index_flatten, [t]]
            s_list[:, :, t] = s_trans.reshape(batch, -1)

    return m_list, s_list, w_list

def MSE(s_list, s, w_list): 
    s_est = np.sum(w_list * s_list, axis=1)
    mse = (s_est - s)**2
    mse_cum = np.cumsum(mse, axis=-1)
    return mse, mse_cum
