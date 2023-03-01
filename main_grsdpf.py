from torch.utils.data import DataLoader
from rspf_dpf import *
from data_generation import *
# import datetime


# start_time = datetime.datetime.now()

def main(): 
    parser = argparse.ArgumentParser(description='Performance Testing')
    parser.add_argument('-de', '--device', dest='device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='choose which device to use')
    parser.add_argument('-nt', '--timestep', dest='N_t', type=int, default=50, help='num of time steps')
    parser.add_argument('-nm', '--modelnum', dest='N_m', type=int, default=8, help='num of models')
    parser.add_argument('-nptr', '--trainparticlenum', dest='N_ptr', type=int, default=200, help='num of particles for training')
    parser.add_argument('-npte', '--testparticlenum', dest='N_pte', type=int, default=2000, help='num of particles for testing')
    parser.add_argument('-mu', '--muu', dest='mu_u', type=float, default=0., help='mean of noise u')
    parser.add_argument('-su', '--sigmau', dest='sigma_u', type=float, default=0.1**0.5, help='standard deviation of noise u')
    parser.add_argument('-mv', '--muv', dest='mu_v', type=float, default=0., help='mean of noise v')
    parser.add_argument('-sv', '--sigmav', dest='sigma_v', type=float, default=0.1**0.5, help='standard deviation of noise v')
    parser.add_argument('-re', '--resample', dest='re', type=str, default='mul', choices=['mul', 'sys'], help='resampling method')
    parser.add_argument('-dy', '--dynamics', action='append', dest='dyn', type=str, default=[], choices=['Mark', 'Poly'], help='model switching dynamics')
    parser.add_argument('-pr', '--proposal', action='append', dest='prop', type=str, default=[], choices=['Boot', 'Uni', 'Deter'], help='proposal type of model index sampling')
    parser.add_argument('-ga', '--gamma', action='append', dest='gamma', type=float, default=[], help='hyperparameter gamma for multi-model particle filter')
    parser.add_argument('-nn', '--neuralnetwork', action='store_false', dest='nn', help='whether use neural network')
    parser.add_argument('-pf', '--particlefilter', action='store_false', dest='pf', help='whether test regime switching particle filter')
    parser.add_argument('-mmpf', '--multiparticlefilter', action='store_false', dest='mmpf', help='whether test multi-model particle filter')
    parser.add_argument('-rs', '--regimeswitching', action='store_false', dest='rs', help='whether operate regime switching dpf')
    parser.add_argument('-si', '--singlemodel', action='store_false', dest='sin', help='whether operate single model dpf')
    parser.add_argument('-nf', '--normalisingflow', action='store_true', dest='nf', help='whether operate normalising flow')
    parser.add_argument('-tr', '--train', action='store_true', dest='train', help='whether train dpf')
    parser.add_argument('-te', '--test', action='store_false', dest='test', help='whether test performance')
    parser.add_argument('-ep', '--epochnum', dest='epoch', type=int, default=60, help='epoch num for training')
    parser.add_argument('-lr', '--learningrate', dest='lr', type=float, default='2e-2', help='learning rate for training')
    parser.add_argument('-ttr', '--trainingtrajectorynum', dest='N_train', type=int, default=1000, help='num of training trajectories')
    parser.add_argument('-tb', '--trainingbatchsize', dest='batch_train', type=int, default=100, help='training batch size')
    parser.add_argument('-vb', '--valbatchsize', dest='batch_val', type=int, default=500, help='validation batch size')
    parser.add_argument('-teb', '--testbatchsize', dest='batch_test', type=int, default=500, help='testing batch size')

    args = parser.parse_args()

    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') 
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    device = torch.device(args.device)
    args_model = create_parameters(args.N_m, device)

    if args.train: 
        for dyn in args.dyn: 
            trainingset = LoadTrainSet(f'dyn{dyn}_mu{args.mu_u}_su{args.sigma_u}_mv{args.mu_v}_sv{args.sigma_v}', args.N_train)
            train_data = DataLoader(dataset=trainingset, batch_size=args.batch_train, shuffle=True)
            validationset = LoadValSet(f'dyn{dyn}_mu{args.mu_u}_su{args.sigma_u}_mv{args.mu_v}_sv{args.sigma_v}', args.N_train)
            val_data = DataLoader(dataset=validationset, batch_size=args.batch_val, shuffle=True)
            if args.sin: 
                dpf = DPF(args.nn, args.nf, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                dpf.train(train_data, val_data, args.epoch, args.N_ptr, dyn, args.re)

            if args.rs: 
                for prop in args.prop: 
                    rsdpf = RSDPF(args.N_m, args.nn, args.nf, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                    rsdpf.train(train_data, val_data, args.epoch, args.N_ptr, dyn, prop, args.re)

    if args.test: 
        for dyn in args.dyn: 
            testset = LoadTestSet(f'dyn{dyn}_mu{args.mu_u}_su{args.sigma_u}_mv{args.mu_v}_sv{args.sigma_v}')
            test_data = DataLoader(dataset=testset, batch_size=args.batch_test)
            
            if args.sin: 
                dpf_test = DPF(args.nn, args.nf, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                dpf_test.test(test_data, args.N_pte, dyn, args.re)
            if args.mmpf: 
                for gamma in args.gamma: 
                    mmpf = MMPF(args.N_m, args_model, args.mu_u, args.sigma_u, args.mu_v, args.sigma_v, gamma, device)
                    mmpf.test(test_data, args.N_pte, dyn, args.re)

            for prop in args.prop: 
                if args.rs: 
                    rsdpf_test = RSDPF(args.N_m, args.nn, args.nf, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                    rsdpf_test.test(test_data, args.N_pte, dyn, prop, args.re)
                if args.pf: 
                    rspf = RSPF(args.N_m, args.mu_u, args.sigma_u, args.mu_v, args.sigma_v, args_model, device)
                    rspf.test(test_data, args.N_pte, dyn, prop, args.re)


if __name__ == '__main__': 
    main()

# paracre_time = datetime.datetime.now()

# dataload_time = datetime.datetime.now()
# dyn_l = ["Mark", "Poly"]
# for dyn in dyn_l: 
#     trainingset = LoadTrainSet(dir=f"{dyn}_0.1")
#     train_data = DataLoader(dataset=trainingset, batch_size=100, shuffle=True)
#     validationset = LoadValSet(dir=f"{dyn}_0.1")
#     val_data = DataLoader(dataset=validationset, batch_size=500, shuffle=True)
#     rsdpf = RSDPF(rs=True, nnm=True, nf=False, tran_matrix=P, beta=beta).to(device)
#     rsdpf.train(train_data, val_data, N_iter=60, N_p=N_p_train, dyn=dyn, prop="Boot", re="mul")  
#     testset = LoadTestSet(dir=f"{dyn}_0.1")
#     test_data = DataLoader(dataset=testset, batch_size=500)
#     rsdpf.test(test_data, N_p=N_p_test, dyn=dyn, prop="Boot", re="mul")

# rs_list = [True, False]

# for rs in rs_list: 
#     for dyn in dyn_list: 
#         trainingset = LoadTrainSet(dir=dyn)
#         train_data = DataLoader(dataset=trainingset, batch_size=100, shuffle=True)
#         validationset = LoadValSet(dir=dyn)
#         val_data = DataLoader(dataset=validationset, batch_size=500, shuffle=True)
#         for prop in prop_list: 
#             rsdpf = RSDPF(rs=rs, nf=False, tran_matrix=P, beta=beta).to(device)
#             rsdpf.train(train_data, val_data, N_iter=40, N_p=N_p_train, dyn=dyn, prop=prop, re=re)
#             testset = LoadTestSet(dir=dyn)
#             test_data = DataLoader(dataset=testset, batch_size=500)
#             rsdpf.test(test_data, N_p=N_p_test, dyn=dyn, prop=prop, re=re)

# # training_time = datetime.datetime.now()

# rspf = RSPF(A, B, C, D, tran_matrix=P, beta=beta)
# for dyn in dyn_list: 
#     testset = LoadTestSet(dir=f"{dyn}_0.1")
#     test_data = DataLoader(dataset=testset, batch_size=500)
#     for prop in prop_list: 
#         rspf.test(test_data, N_p=N_p_test, dyn=dyn, prop=prop, re=re)

# for gamma in gamma_list: 
#     mmpf = MMPF(A, B, C, D, gamma=torch.tensor(gamma))
#     for dyn in dyn_list: 
#         testset = LoadTestSet(dir=f"{dyn}_0.1")
#         test_data = DataLoader(dataset=testset, batch_size=500)
#         mmpf.test(test_data, N_p=N_p_test, re=re, dyn=dyn)

# # test_time = datetime.datetime.now()

# # print(f"create parameter time:{(paracre_time - start_time).seconds}, load testdata time: {(dataload_time - paracre_time).seconds}, training time: {(training_time - dataload_time).seconds}, testing time: {(testing_time - traing_time).seconds}")






