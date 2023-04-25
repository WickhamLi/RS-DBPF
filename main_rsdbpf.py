from torch.utils.data import DataLoader
from classes import *
from data_generation import *


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
    parser.add_argument('-rspf', '--regimeswitchingpf', action='store_false', dest='rspf', help='whether test regime switching particle filter')
    parser.add_argument('-mmpf', '--multimodelpf', action='store_false', dest='mmpf', help='whether test multi-model particle filter')
    parser.add_argument('-rsdpf', '--regimeswitchingdpf', action='store_false', dest='rsdpf', help='whether operate regime switching dpf')
    parser.add_argument('-sidpf', '--singlemodeldpf', action='store_false', dest='sin', help='whether operate single model dpf')
    parser.add_argument('-tr', '--train', action='store_false', dest='train', help='whether train dpf')
    parser.add_argument('-te', '--test', action='store_false', dest='test', help='whether test performance')
    parser.add_argument('-ep', '--epochnum', dest='epoch', type=int, default=60, help='epoch num for training')
    parser.add_argument('-lr', '--learningrate', dest='lr', type=float, default='5e-2', help='learning rate for training')
    parser.add_argument('-ttr', '--trainingtrajectorynum', dest='N_train', type=int, default=1000, help='num of training trajectories')
    parser.add_argument('-tb', '--trainingbatchsize', dest='batch_train', type=int, default=100, help='training batch size')
    parser.add_argument('-vb', '--valbatchsize', dest='batch_val', type=int, default=500, help='validation batch size')
    parser.add_argument('-teb', '--testbatchsize', dest='batch_test', type=int, default=500, help='testing batch size')

    args = parser.parse_args()

    device = torch.device(args.device)
    args_model = create_parameters(args.N_m, device)

    if args.train: 
        for dyn in args.dyn: 
            trainingset = LoadTrainSet(f'{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}', args.N_train)
            train_data = DataLoader(dataset=trainingset, batch_size=args.batch_train, shuffle=True)
            validationset = LoadValSet(f'{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}', args.N_train)
            val_data = DataLoader(dataset=validationset, batch_size=args.batch_val, shuffle=True)
            if args.sin: 
                dpf = DPF(args.nn, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                dpf.train(train_data, val_data, args.epoch, args.N_ptr, dyn, args.re)

            if args.rsdpf: 
                for prop in args.prop: 
                    rsdpf = RSDPF(args.N_m, args.nn, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                    rsdpf.train(train_data, val_data, args.epoch, args.N_ptr, dyn, prop, args.re)

    if args.test: 
        for dyn in args.dyn: 
            testset = LoadTestSet(f'{dyn}_mu{args.mu_u}_su{args.sigma_u:.4f}_mv{args.mu_v}_sv{args.sigma_v:.4f}')
            test_data = DataLoader(dataset=testset, batch_size=args.batch_test)
            
            if args.sin: 
                dpf_test = DPF(args.nn, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                dpf_test.test(test_data, args.N_pte, dyn, args.re)
            if args.mmpf: 
                for gamma in args.gamma: 
                    mmpf = MMPF(args.N_m, args_model, args.mu_u, args.sigma_u, args.mu_v, args.sigma_v, gamma, device)
                    mmpf.test(test_data, args.N_pte, dyn, args.re)

            for prop in args.prop: 
                if args.rsdpf: 
                    rsdpf_test = RSDPF(args.N_m, args.nn, args.mu_u, args.mu_v, args_model, args.lr, device).to(device)
                    rsdpf_test.test(test_data, args.N_pte, dyn, prop, args.re)
                if args.rspf: 
                    rspf = RSPF(args.N_m, args.mu_u, args.sigma_u, args.mu_v, args.sigma_v, args_model, device)
                    rspf.test(test_data, args.N_pte, dyn, prop, args.re)


if __name__ == '__main__': 
    main()





