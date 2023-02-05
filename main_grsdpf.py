# from rspf_np import *
from rspf_torch import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

P, A, B, C, D, beta = create_parameters()
rspf = RSPF(P, A, B, C, D, beta=beta)
    
T = 50
N_p = 2000
run = 1000

m, s, o = generate_data(T, rspf, batch=run, dyn="Mark")

# m_data.to(device)
# s_data.to(device)
# o_data.to(device)

m_np = m.numpy().reshape(run, T)
m_df = pd.DataFrame(m_np)
m_train, m_test = train_test_split(m_df, test_size=0.05, shuffle=False)
s_np = s.numpy().reshape(run, T)
s_df = pd.DataFrame(s_np)
s_train, s_test = train_test_split(s_df, test_size=0.05, shuffle=False)
o_np = o.numpy().reshape(run, T)
o_df = pd.DataFrame(o_np)
o_train, o_test = train_test_split(o_df, test_size=0.05, shuffle=False)

m_train.to_csv('m_train')
s_train.to_csv('s_train')
o_train.to_csv('o_train')

m_test.to_csv('m_test')
s_test.to_csv('s_test')
o_test.to_csv('o_test')


class LoadTrainSet(Dataset): 
    def __init__(self): 
        m_data = np.loadtxt('./m_train', delimiter=',', skiprows=1)
        self.m = torch.from_numpy(m_data[:900, np.newaxis, 1:])
        s_data = np.loadtxt('./s_train', delimiter=',', skiprows=1)
        self.s = torch.from_numpy(s_data[:900, np.newaxis, 1:])
        o_data = np.loadtxt('./o_train', delimiter=',', skiprows=1)
        self.o = torch.from_numpy(o_data[:900, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadValSet(Dataset): 
    def __init__(self): 
        m_data = np.loadtxt('./m_train', delimiter=',', skiprows=1)
        self.m = torch.from_numpy(m_data[900:, np.newaxis, 1:])
        s_data = np.loadtxt('./s_train', delimiter=',', skiprows=1)
        self.s = torch.from_numpy(s_data[900:, np.newaxis, 1:])
        o_data = np.loadtxt('./o_train', delimiter=',', skiprows=1)
        self.o = torch.from_numpy(o_data[900:, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

class LoadTestSet(Dataset): 
    def __init__(self): 
        m_data = np.loadtxt('./m_test', delimiter=',', skiprows=1)
        self.m = torch.from_numpy(m_data[:, np.newaxis, 1:])
        s_data = np.loadtxt('./s_test', delimiter=',', skiprows=1)
        self.s = torch.from_numpy(s_data[:, np.newaxis, 1:])
        o_data = np.loadtxt('./o_test', delimiter=',', skiprows=1)
        self.o = torch.from_numpy(o_data[:, np.newaxis, 1:])
        self.nums = self.m.shape[0]
        
    def __getitem__(self, index): 
        return self.m[index], self.s[index], self.o[index]

    def __len__(self): 
        return self.nums

trainingset = LoadTrainSet()
train_data = DataLoader(dataset=trainingset, batch_size=50, shuffle=True)
validationset = LoadValSet()
val_data = DataLoader(dataset=validationset, batch_size=50, shuffle=True)


# # dyn=Mark/Poly, prop=Boot/Uni/Deter, re=sys/mul
rsdpf = RSDPF(P, beta=beta)
rsdpf.training(train_data, val_data, dyn="Mark")

testset = LoadTestSet()
test_data = DataLoader(dataset=testset, batch_size=50, shuffle=True)
rsdpf.testing(test_data, A, B, C, D, dyn="Mark")


