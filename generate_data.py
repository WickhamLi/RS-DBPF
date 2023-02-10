from sklearn.model_selection import train_test_split
from rspf_dpf import *

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  
# device = torch.device('cpu')

def create_parameters(N_m=8): 
    P = torch.zeros(N_m, N_m, device=device)
    for i in range(N_m): 
        if not i: 
            P += torch.diag(torch.tensor([0.80]*N_m, device=device), i)
        elif i==1: 
            P += torch.diag(torch.tensor([0.15]*(8-i), device=device), i)
            P += torch.diag(torch.tensor([0.15]*i, device=device), i-8)
        else: 
            P += torch.diag(torch.tensor([1/120]*(8-i), device=device), i)
            P += torch.diag(torch.tensor([1/120]*i, device=device), i-8)

    A = torch.tensor([-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9], device=device)
    B = torch.tensor([0., -2., 2., -4., 0., 2., -2., 4.], device=device)
    C = A.clone()
    D = B.clone()
    beta = torch.tensor([1]*N_m, device=device)
    return P, A, B, C, D, beta

def data_generation(T, model, batch=1, dyn="Mark"): 
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

        
# T = 50
# run = 2000
# dyn = "Mark"

# P, A, B, C, D, beta = create_parameters(N_m=8)

# rspf_simu = RSPF(A, B, C, D, tran_matrix=P, beta=beta)

# m, s, o = data_generation(T, rspf_simu, batch=run, dyn=dyn)

# m_np = m.numpy().reshape(run, T)
# m_df = pd.DataFrame(m_np, dtype=np.float32)
# m_train, m_test = train_test_split(m_df, test_size=0.25, shuffle=False)
# s_np = s.numpy().reshape(run, T)
# s_df = pd.DataFrame(s_np, dtype=np.float32)
# s_train, s_test = train_test_split(s_df, test_size=0.25, shuffle=False)
# o_np = o.numpy().reshape(run, T)
# o_df = pd.DataFrame(o_np, dtype=np.float32)
# o_train, o_test = train_test_split(o_df, test_size=0.25, shuffle=False)

# m_train.to_csv(f'./datasets/{dyn}/m_train')
# s_train.to_csv(f'./datasets/{dyn}/s_train')
# o_train.to_csv(f'./datasets/{dyn}/o_train')

# m_test.to_csv(f'./datasets/{dyn}/m_test')
# s_test.to_csv(f'./datasets/{dyn}/s_test')
# o_test.to_csv(f'./datasets/{dyn}/o_test')

