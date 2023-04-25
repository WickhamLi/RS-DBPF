import torch
import torch.nn as nn

# Neural network for DBPF
class dynamic_NN(nn.Module): 
    def __init__(self, input, hidden, output): 
        super().__init__()
        self.input = input
        self.fc = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))

    def forward(self, s): 
        return torch.squeeze(self.fc(s.view(-1, self.input)))

# Neural network for RS-DBPF
class dynamic_RSNN(nn.Module): 
    def __init__(self, input, hidden, output): 
        super().__init__()
        self.input = input
        self.fc0 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))
        self.fc1 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))
        self.fc2 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))
        self.fc3 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))
        self.fc4 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))
        self.fc5 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))
        self.fc6 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))
        self.fc7 = nn.Sequential(nn.Linear((input), (hidden), dtype=torch.float32), nn.Tanh(), nn.Linear((hidden), (output), dtype=torch.float32))

    def forward(self, s, m): 
        if m==0: 
            out = self.fc0(s.view(-1, self.input))
        elif m==1: 
            out = self.fc1(s.view(-1, self.input))
        elif m==2: 
            out = self.fc2(s.view(-1, self.input))
        elif m==3: 
            out = self.fc3(s.view(-1, self.input))
        elif m==4: 
            out = self.fc4(s.view(-1, self.input))
        elif m==5: 
            out = self.fc5(s.view(-1, self.input))
        elif m==6: 
            out = self.fc6(s.view(-1, self.input))
        elif m==7: 
            out = self.fc7(s.view(-1, self.input))
        return torch.squeeze(out)


