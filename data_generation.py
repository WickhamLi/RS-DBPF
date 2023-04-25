import argparse
import os
from sklearn.model_selection import train_test_split
from classes import *
from utils import *

def create_parameters(N_m, device): 
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
    C = A.clone().detach()
    D = B.clone().detach()
    beta = torch.tensor([1.]*N_m, device=device)
    return P, A, B, C, D, beta

def data_generation(T, batch, dyn, model): 
    m = torch.zeros(batch, 1, T, dtype=torch.long, device=model.device)
    s = torch.zeros(batch, 1, T, device=model.device)
    o = torch.zeros(batch, 1, T, device=model.device)
    m[:, :, 0], s[:, :, 0] = model.initial(size=(batch, 1))
    o[:, :, 0] = model.obs(m[:, :, 0], s[:, :, 0])
    for t in range(1, T):
        if dyn=="Poly": 
            m[:, :, t] = model.Polyaurn_dynamic(m[:, :, :t])
        else: 
            m[:, :, t] = model.Markov_dynamic(m[:, :, t-1])
        s[:, :, t] = model.state(m[:, :, t], s[:, :, t-1])
        o[:, :, t] = model.obs(m[:, :, t], s[:, :, t])
    if model.device == torch.device('cpu'): 
        return m.squeeze().numpy(), s.squeeze().numpy(), o.squeeze().numpy()
    else: 
        return m.squeeze().cpu().numpy(), s.squeeze().cpu().numpy(), o.squeeze().cpu().numpy()

def main(): 
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('-de', '--device', dest='device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='choose which device to use')
    parser.add_argument('-nt', '--timestep', dest='N_t', type=int, default=50, help='num of time steps')
    parser.add_argument('-tr', '--trajectorynum', dest='N_tra', type=int, default=2000, help='num of trajectories')
    parser.add_argument('-nm', '--modelnum', dest='N_m', type=int, default=8, help='num of models')
    parser.add_argument('-mu', '--muu', dest='mu_u', type=float, default=0., help='mean of noise u')
    parser.add_argument('-su', '--sigmau', dest='sigma_u', type=float, default=0.1**0.5, help='standard deviation of noise u')
    parser.add_argument('-mv', '--muv', dest='mu_v', type=float, default=0., help='mean of noise v')
    parser.add_argument('-sv', '--sigmav', dest='sigma_v', type=float, default=0.1**0.5, help='standard deviation of noise v')
    parser.add_argument('-ts', '--testsize', dest='test_size', type=float, default=0.25, help='percentage of test datasets')
    parser.add_argument('-dy', '--dynamics', action='append', dest='dyn', type=str, choices=['Mark', 'Poly'], help='model switching dynamics', required=True)

    args = parser.parse_args()

    device = torch.device(args.device)

    args_model = create_parameters(args.N_m, device)
    rspf_simu = RSPF(args.N_m, args.mu_u, args.sigma_u, args.mu_v, args.sigma_v, args_model, device)

    if not os.path.isdir('datasets'): 
        os.mkdir('datasets')

    for dyn in args.dyn: 
        dyn_dir = os.path.join('datasets', f'{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}')
        if not os.path.isdir(dyn_dir): 
            os.mkdir(dyn_dir)        

        m, s, o = data_generation(args.N_t, args.N_tra, dyn, rspf_simu)


        m_df = pd.DataFrame(m, dtype=np.float32)
        m_train, m_test = train_test_split(m_df, test_size=args.test_size, shuffle=False)

        s_df = pd.DataFrame(s, dtype=np.float32)
        s_train, s_test = train_test_split(s_df, test_size=args.test_size, shuffle=False)

        o_df = pd.DataFrame(o, dtype=np.float32)
        o_train, o_test = train_test_split(o_df, test_size=args.test_size, shuffle=False)

        m_train.to_csv(f'./datasets/{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}/m_train')
        s_train.to_csv(f'./datasets/{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}/s_train')
        o_train.to_csv(f'./datasets/{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}/o_train')

        m_test.to_csv(f'./datasets/{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}/m_test')
        s_test.to_csv(f'./datasets/{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}/s_test')
        o_test.to_csv(f'./datasets/{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}/o_test')

if __name__ == '__main__': 
    main()
