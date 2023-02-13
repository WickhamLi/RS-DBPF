from sklearn.model_selection import train_test_split
from rspf_dpf import *
from utils import *

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')  
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
        
T = 50
run = 2000
dyn = ["Mark", "Poly"]

P, A, B, C, D, beta = create_parameters(N_m=8)

rspf_simu = RSPF(A, B, C, D, tran_matrix=P, beta=beta)

for dyn in dyn: 
    m, s, o = data_generation(T, rspf_simu, batch=run, dyn=dyn)

    m_np = m.numpy().reshape(run, T)
    m_df = pd.DataFrame(m_np, dtype=np.float32)
    m_train, m_test = train_test_split(m_df, test_size=0.25, shuffle=False)
    s_np = s.numpy().reshape(run, T)
    s_df = pd.DataFrame(s_np, dtype=np.float32)
    s_train, s_test = train_test_split(s_df, test_size=0.25, shuffle=False)
    o_np = o.numpy().reshape(run, T)
    o_df = pd.DataFrame(o_np, dtype=np.float32)
    o_train, o_test = train_test_split(o_df, test_size=0.25, shuffle=False)

    m_train.to_csv(f'./datasets/{dyn}/m_train')
    s_train.to_csv(f'./datasets/{dyn}/s_train')
    o_train.to_csv(f'./datasets/{dyn}/o_train')

    m_test.to_csv(f'./datasets/{dyn}/m_test')
    s_test.to_csv(f'./datasets/{dyn}/s_test')
    o_test.to_csv(f'./datasets/{dyn}/o_test')

