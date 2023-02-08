import torch
import torch.nn as nn


class PlanarTrans(nn.Module): 
    def __init__(self, s_dim=1, N_m=8):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(N_m, s_dim).normal_(0, 0.1))
        # self.w.squeeze_()
        self.b = nn.Parameter(torch.Tensor(N_m).normal_(0, 0.1))
        self.u = nn.Parameter(torch.Tensor(N_m, s_dim).normal_(0, 0.1))
        # self.u.squeeze_()

    def foward(self, m, s): 
        if ((self.u[m] * self.w[m]).sum(dim=-1, keepdim=True) < -1).cum(): 
            # self.modified_u(m)
            wu = (self.u[m] * self.w[m]).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            self.u[m].data.where((self.u[m] * self.w[m]).sum(dim=-1, keepdim=True)<-1, (self.u[m] + (m - wu) * self.w[m] / torch.norm(self.w[m], p=2, dim=-1) ** 2))

        return s + self.u[m] * torch.tanh((self.w[m] * s).sum(dim=-1, keepdim=True) + self.b[m])

    def log_detJ(self, m, s): 
        if ((self.u[m] * self.w[m]).sum(dim=-1, keepdim=True) < -1).cum(): 
            # self.modified_u(m)
            wu = (self.u[m] * self.w[m]).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            self.u[m].data.where((self.u[m] * self.w[m]).sum(dim=-1, keepdim=True)<-1, (self.u[m] + (m - wu) * self.w[m] / torch.norm(self.w[m], p=2, dim=-1) ** 2))
        
        phi = 1 - torch.tanh((self.w[m] * s).sum(dim=-1, keepdim=True) + self.b[m])**2 * self.w[m]
        abs_detJ = torch.abs(1 + (self.u[m] * phi).sum(dim=-1, keepdim=True))

        return torch.log(1e-4 + abs_detJ)  

    # def modified_u(self, m): 
    #     wu = self.w[m].T @ self.u[m]
    #     m = -1 + torch.log(1 + torch.exp(wu))
    #     self.u[m].data = (self.u[m] + (m - wu) * self.w[m] / torch.norm(self.w[m], p=2, dim=0) ** 2)


class PlanarFlows(nn.Module): 
    def __init__(self, s_dim=1, N_m=8, layer=2):
        super().__init__()
        self.layers = [PlanarTrans(s_dim, N_m) for _ in range(layer)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, m, s): 
        log_detJ = 0. 

        for flow in self.layers: 
            log_detJ += flow.log_detJ(m, s)
            s = flow(m, s)

        return s, log_detJ




class Cond_PlanarTrans(nn.Module): 
    def __init__(self, o_dim=1, N_m=8):
        super().__init__()
        self.dim = o_dim
        self.N_m = N_m
        self.fc1 = nn.Sequential(nn.Linear((o_dim), (N_m * o_dim), dtype=torch.float32), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear((o_dim), (N_m * o_dim), dtype=torch.float32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear((o_dim), (N_m), dtype=torch.float32), nn.ReLU())

    def forward(self, m, s, o): 
        w_pre = self.fc1(o)
        # w.squeeze_()
        u_pre = self.fc2(o)      
        # u.squeeze_()
        b_pre = self.fc3(o)

        if w_pre.shape[1] < m.shape[1]: 
            w_tile = w_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
            u_tile = u_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
            b_tile = b_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        else: 
            w_tile = w_pre
            u_tile = u_pre
            b_tile = b_pre

        if s.shape[1] < m.shape[1]: 
            s_tile = s.tile(1, int(m.shape[1]/s.shape[1]), 1)
        else: 
            s_tile = s

        if w_tile.shape[-1] > self.N_m: 
            batch, num, time = o.shape
            w_tile = w_tile.view(batch, num, self.N_m, self.o_dim)
            u_tile = u_tile.view(batch, num, self.N_m, self.o_dim)

        batch_index = tuple(i//m.shape[1] for i in range(m.shape[0] * m.shape[1]))
        pars_index = tuple(torch.arange(m.shape[1]).tile(m.shape[0]))
        index = tuple(m.view(-1))
        w = w_tile[batch_index, pars_index, index].view(m.shape)
        u = u_tile[batch_index, pars_index, index].view(m.shape)
        b = b_tile[batch_index, pars_index, index].view(m.shape)


        if ((u * w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (u * w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            u.where((u * w).sum(dim=-1, keepdim=True)<-1, (u + (m - wu) * w / torch.norm(w, p=2, dim=-1) ** 2))

        return s_tile + u * torch.tanh((w * s_tile).sum(dim=-1, keepdim=True) + b)

    def log_detJ(self, m, s, o): 
        w_pre = self.fc1(o)
        # w = w.view(self.N_m, self.o_dim)
        # w.squeeze_()
        u_pre = self.fc2(o)
        # u = u.view(self.N_m, self.o_dim)
        # u.squeeze_()
        b_pre = self.fc3(o)

        if w_pre.shape[1] < m.shape[1]: 
            w_tile = w_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
            u_tile = u_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
            b_tile = b_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        else: 
            w_tile = w_pre
            u_tile = u_pre
            b_tile = b_pre

        if s.shape[1] < m.shape[1]: 
            s_tile = s.tile(1, int(m.shape[1]/s.shape[1]), 1)
        else: 
            s_tile = s
        
        if w_tile.shape[-1] > self.N_m: 
            batch, num, time = o.shape
            w_tile = w_tile.view(batch, num, self.N_m, self.o_dim)
            u_tile = u_tile.view(batch, num, self.N_m, self.o_dim)

        batch_index = tuple(i//m.shape[1] for i in range(m.shape[0] * m.shape[1]))
        pars_index = tuple(torch.arange(m.shape[1]).tile(m.shape[0]))
        index = tuple(m.view(-1))
        w = w_tile[batch_index, pars_index, index]
        u = u_tile[batch_index, pars_index, index]
        b = b_tile[batch_index, pars_index, index]

        if ((u * w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (u * w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            u.where((u * w).sum(dim=-1, keepdim=True)<-1, (u + (m - wu) * w / torch.norm(w, p=2, dim=-1) ** 2))

        phi = 1 - torch.tanh((w * s_tile).sum(dim=-1, keepdim=True) + b)**2 * w
        abs_detJ = torch.abs(1 + (u * phi).sum(dim=-1, keepdim=True))

        return torch.log(1e-4 + abs_detJ)

    # def modified_u(self, w, u): 
    #     wu = w.T @ u
    #     m = -1 + torch.log(1 + torch.exp(wu))

    #     return u + (m - wu) * w / torch.norm(w, p=2, dim=0)**2


class Cond_PlanarFlows(nn.Module): 
    def __init__(self, dim=1, N_m=8, layer=2):
        super().__init__()
        self.layers = [Cond_PlanarTrans(dim, N_m) for _ in range(layer)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, m, s, o): 
        log_detJ = 0. 

        for flow in self.layers: 
            log_detJ += flow.log_detJ(m, s, o)
            s = flow(m, s, o)
        
        return s, log_detJ