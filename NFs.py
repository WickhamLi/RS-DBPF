import torch
import torch.nn as nn


class PlanarTrans(nn.Module): 
    def __init__(self, s_dim=1, N_m=8):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(N_m, s_dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.Tensor(N_m, 1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.Tensor(N_m, s_dim).normal_(0, 0.1))

    def foward(self, m, s): 
        if ((self.u * self.w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (self.u * self.w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            self.u.data = torch.where((self.u * self.w).sum(dim=-1, keepdim=True)<-1, (self.u + (m - wu) * self.w / torch.norm(self.w, p=2, dim=-1) ** 2), self.u.data)

        return s + self.u[m] * torch.tanh((self.w[m] * s).sum(dim=-1, keepdim=True) + self.b[m])

    def log_detJ(self, m, s): 
        if ((self.u * self.w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (self.u * self.w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            self.u.data = torch.where((self.u * self.w).sum(dim=-1, keepdim=True)<-1, (self.u + (m - wu) * self.w / torch.norm(self.w, p=2, dim=-1) ** 2), self.u.data)
        
        phi = 1 - torch.tanh((self.w[m] * s).sum(dim=-1, keepdim=True) + self.b[m])**2 * self.w[m]
        abs_detJ = torch.abs(1 + (self.u[m] * phi).sum(dim=-1))

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

    def forward(self, m, s_t, o): 
        w = torch.empty(self.N_m, self.dim)
        u = torch.empty(self.N_m, self.dim)
        b = torch.empty(self.N_m, 1)
        
        w_pre = self.fc1(o)
        w = w_pre.view(o.shape[0], self.N_m, self.dim)
        u_pre = self.fc2(o)      
        u = u_pre.view(o.shape[0], self.N_m, self.dim)
        b_pre = self.fc3(o)
        b = b_pre.view(o.shape[0], self.N_m, 1)

        if ((u * w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (u * w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            index_1, index_2 = torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[0], torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[1]
            u[index_1, index_2, :] = u[index_1, index_2, :] + (m[index_1, index_2, :] - wu[index_1, index_2, :]) * w[index_1, index_2] / torch.norm(w[index_1, index_2, :], p=2, dim=-1) ** 2

        s_p = torch.empty(s_t.shape)
        index_1, index_2 = torch.where(m==0)[0], torch.where(m==0)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 0, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 0, :]).sum(dim=-1, keepdim=True) + b[index_1, 0, :])
        
        index_1, index_2 = torch.where(m==1)[0], torch.where(m==1)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 1, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 1, :]).sum(dim=-1, keepdim=True) + b[index_1, 1, :])

        index_1, index_2 = torch.where(m==2)[0], torch.where(m==2)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 2, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 2, :]).sum(dim=-1, keepdim=True) + b[index_1, 2, :])

        index_1, index_2 = torch.where(m==3)[0], torch.where(m==3)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 3, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 3, :]).sum(dim=-1, keepdim=True) + b[index_1, 3, :])

        index_1, index_2 = torch.where(m==4)[0], torch.where(m==4)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 4, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 4, :]).sum(dim=-1, keepdim=True) + b[index_1, 4, :])

        index_1, index_2 = torch.where(m==5)[0], torch.where(m==5)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 5, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 5, :]).sum(dim=-1, keepdim=True) + b[index_1, 5, :])

        index_1, index_2 = torch.where(m==6)[0], torch.where(m==6)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 6, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 6, :]).sum(dim=-1, keepdim=True) + b[index_1, 6, :])

        index_1, index_2 = torch.where(m==7)[0], torch.where(m==7)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 7, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 7, :]).sum(dim=-1, keepdim=True) + b[index_1, 7, :])


        # if w_pre.shape[1] < m.shape[1]: 
        #     w_tile = w_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     u_tile = u_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     b_tile = b_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        # else: 
        #     w_tile = w_pre
        #     u_tile = u_pre
        #     b_tile = b_pre

        # if s.shape[1] < m.shape[1]: 
        #     s_tile = s.tile(1, int(m.shape[1]/s.shape[1]), 1)
        # else: 
        #     s_tile = s

        # if w_tile.shape[-1] > self.N_m: 
        #     batch, num, time = o.shape
        #     w_tile = w_tile.view(batch, num, self.N_m, self.o_dim)
        #     u_tile = u_tile.view(batch, num, self.N_m, self.o_dim)

        # batch_index = tuple(i//m.shape[1] for i in range(m.shape[0] * m.shape[1]))
        # pars_index = tuple(torch.arange(m.shape[1]).tile(m.shape[0]))
        # index = tuple(m.view(-1))
        # w = w_tile[batch_index, pars_index, index].view(m.shape)
        # u = u_tile[batch_index, pars_index, index].view(m.shape)
        # b = b_tile[batch_index, pars_index, index].view(m.shape)
        return s_p

    def log_detJ(self, m, s, o): 
        w = torch.empty(self.N_m, self.dim)
        u = torch.empty(self.N_m, self.dim)
        b = torch.empty(self.N_m, 1)
        
        w_pre = self.fc1(o)
        w = w_pre.view(o.shape[0], self.N_m, self.dim)
        u_pre = self.fc2(o)      
        u = u_pre.view(o.shape[0], self.N_m, self.dim)
        b_pre = self.fc3(o)
        b = b_pre.view(o.shape[0], self.N_m, 1)

        # if w_pre.shape[1] < m.shape[1]: 
        #     w_tile = w_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     u_tile = u_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     b_tile = b_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        # else: 
        #     w_tile = w_pre
        #     u_tile = u_pre
        #     b_tile = b_pre

        # if s.shape[1] < m.shape[1]: 
        #     s_tile = s.tile(1, int(m.shape[1]/s.shape[1]), 1)
        # else: 
        #     s_tile = s
        
        # if w_tile.shape[-1] > self.N_m: 
        #     batch, num, time = o.shape
        #     w_tile = w_tile.view(batch, num, self.N_m, self.o_dim)
        #     u_tile = u_tile.view(batch, num, self.N_m, self.o_dim)

        # batch_index = tuple(i//m.shape[1] for i in range(m.shape[0] * m.shape[1]))
        # pars_index = tuple(torch.arange(m.shape[1]).tile(m.shape[0]))
        # index = tuple(m.view(-1))
        # w = w_tile[batch_index, pars_index, index].view(m.shape)
        # u = u_tile[batch_index, pars_index, index].view(m.shape)
        # b = b_tile[batch_index, pars_index, index].view(m.shape)

        if ((u * w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (u * w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            index_1, index_2 = torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[0], torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[1]
            u[[index_1], [index_2], :] = u[[index_1], [index_2], :] + (m[[index_1], [index_2], :] - wu[[index_1], [index_2], :]) * w[[index_1], [index_2]] / torch.norm(w[[index_1], [index_2], :], p=2, dim=-1) ** 2

        phi = torch.empty(s.shape)
        abs_detJ = torch.empty(m.shape[:2])
        index_1, index_2 = torch.where(m==0)[0], torch.where(m==0)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 0, :]).sum(dim=-1, keepdim=True) + b[index_1, 0, :])**2 * w[index_1, 0, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 0, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==1)[0], torch.where(m==1)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 1, :]).sum(dim=-1, keepdim=True) + b[index_1, 1, :])**2 * w[index_1, 1, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 1, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==2)[0], torch.where(m==2)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 2, :]).sum(dim=-1, keepdim=True) + b[index_1, 2, :])**2 * w[index_1, 2, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 2, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==3)[0], torch.where(m==3)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 3, :]).sum(dim=-1, keepdim=True) + b[index_1, 3, :])**2 * w[index_1, 3, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 3, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==4)[0], torch.where(m==4)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 4, :]).sum(dim=-1, keepdim=True) + b[index_1, 4, :])**2 * w[index_1, 4, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 4, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==5)[0], torch.where(m==5)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 5, :]).sum(dim=-1, keepdim=True) + b[index_1, 5, :])**2 * w[index_1, 5, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 5, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==6)[0], torch.where(m==6)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 6, :]).sum(dim=-1, keepdim=True) + b[index_1, 6, :])**2 * w[index_1, 6, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 6, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==7)[0], torch.where(m==7)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 7, :]).sum(dim=-1, keepdim=True) + b[index_1, 7, :])**2 * w[index_1, 7, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 7, :]).sum(dim=-1))

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


class CondPlanarTrans_Measurement(nn.Module): 
    def __init__(self, o_dim=1, N_m=8):
        super().__init__()
        self.dim = o_dim
        self.N_m = N_m
        self.fc1 = nn.Sequential(nn.Linear((o_dim), (N_m * o_dim), dtype=torch.float32), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear((o_dim), (N_m * o_dim), dtype=torch.float32), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear((o_dim), (N_m), dtype=torch.float32), nn.ReLU())

    def forward(self, m, s_t, o): 
        w = torch.empty(self.N_m, self.dim)
        u = torch.empty(self.N_m, self.dim)
        b = torch.empty(self.N_m, 1)
        
        w_pre = self.fc1(o)
        w = w_pre.view(o.shape[0], self.N_m, self.dim)
        u_pre = self.fc2(o)      
        u = u_pre.view(o.shape[0], self.N_m, self.dim)
        b_pre = self.fc3(o)
        b = b_pre.view(o.shape[0], self.N_m, 1)

        if ((u * w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (u * w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            index_1, index_2 = torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[0], torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[1]
            u[index_1, index_2, :] = u[index_1, index_2, :] + (m[index_1, index_2, :] - wu[index_1, index_2, :]) * w[index_1, index_2] / torch.norm(w[index_1, index_2, :], p=2, dim=-1) ** 2

        s_p = torch.empty(s_t.shape)
        index_1, index_2 = torch.where(m==0)[0], torch.where(m==0)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 0, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 0, :]).sum(dim=-1, keepdim=True) + b[index_1, 0, :])
        
        index_1, index_2 = torch.where(m==1)[0], torch.where(m==1)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 1, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 1, :]).sum(dim=-1, keepdim=True) + b[index_1, 1, :])

        index_1, index_2 = torch.where(m==2)[0], torch.where(m==2)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 2, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 2, :]).sum(dim=-1, keepdim=True) + b[index_1, 2, :])

        index_1, index_2 = torch.where(m==3)[0], torch.where(m==3)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 3, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 3, :]).sum(dim=-1, keepdim=True) + b[index_1, 3, :])

        index_1, index_2 = torch.where(m==4)[0], torch.where(m==4)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 4, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 4, :]).sum(dim=-1, keepdim=True) + b[index_1, 4, :])

        index_1, index_2 = torch.where(m==5)[0], torch.where(m==5)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 5, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 5, :]).sum(dim=-1, keepdim=True) + b[index_1, 5, :])

        index_1, index_2 = torch.where(m==6)[0], torch.where(m==6)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 6, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 6, :]).sum(dim=-1, keepdim=True) + b[index_1, 6, :])

        index_1, index_2 = torch.where(m==7)[0], torch.where(m==7)[1]
        s_p[index_1, index_2, :] = s_t[index_1, index_2, :] + u[index_1, 7, :] * torch.tanh((s_t[index_1, index_2, :] * w[index_1, 7, :]).sum(dim=-1, keepdim=True) + b[index_1, 7, :])


        # if w_pre.shape[1] < m.shape[1]: 
        #     w_tile = w_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     u_tile = u_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     b_tile = b_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        # else: 
        #     w_tile = w_pre
        #     u_tile = u_pre
        #     b_tile = b_pre

        # if s.shape[1] < m.shape[1]: 
        #     s_tile = s.tile(1, int(m.shape[1]/s.shape[1]), 1)
        # else: 
        #     s_tile = s

        # if w_tile.shape[-1] > self.N_m: 
        #     batch, num, time = o.shape
        #     w_tile = w_tile.view(batch, num, self.N_m, self.o_dim)
        #     u_tile = u_tile.view(batch, num, self.N_m, self.o_dim)

        # batch_index = tuple(i//m.shape[1] for i in range(m.shape[0] * m.shape[1]))
        # pars_index = tuple(torch.arange(m.shape[1]).tile(m.shape[0]))
        # index = tuple(m.view(-1))
        # w = w_tile[batch_index, pars_index, index].view(m.shape)
        # u = u_tile[batch_index, pars_index, index].view(m.shape)
        # b = b_tile[batch_index, pars_index, index].view(m.shape)
        return s_p

    def log_detJ(self, m, s, o): 
        w = torch.empty(self.N_m, self.dim)
        u = torch.empty(self.N_m, self.dim)
        b = torch.empty(self.N_m, 1)
        
        w_pre = self.fc1(o)
        w = w_pre.view(o.shape[0], self.N_m, self.dim)
        u_pre = self.fc2(o)      
        u = u_pre.view(o.shape[0], self.N_m, self.dim)
        b_pre = self.fc3(o)
        b = b_pre.view(o.shape[0], self.N_m, 1)

        # if w_pre.shape[1] < m.shape[1]: 
        #     w_tile = w_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     u_tile = u_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        #     b_tile = b_pre.tile(1, int(m.shape[1]/w_pre.shape[1]), 1)
        # else: 
        #     w_tile = w_pre
        #     u_tile = u_pre
        #     b_tile = b_pre

        # if s.shape[1] < m.shape[1]: 
        #     s_tile = s.tile(1, int(m.shape[1]/s.shape[1]), 1)
        # else: 
        #     s_tile = s
        
        # if w_tile.shape[-1] > self.N_m: 
        #     batch, num, time = o.shape
        #     w_tile = w_tile.view(batch, num, self.N_m, self.o_dim)
        #     u_tile = u_tile.view(batch, num, self.N_m, self.o_dim)

        # batch_index = tuple(i//m.shape[1] for i in range(m.shape[0] * m.shape[1]))
        # pars_index = tuple(torch.arange(m.shape[1]).tile(m.shape[0]))
        # index = tuple(m.view(-1))
        # w = w_tile[batch_index, pars_index, index].view(m.shape)
        # u = u_tile[batch_index, pars_index, index].view(m.shape)
        # b = b_tile[batch_index, pars_index, index].view(m.shape)

        if ((u * w).sum(dim=-1, keepdim=True) < -1).sum(): 
            # self.modified_u(m)
            wu = (u * w).sum(dim=-1, keepdim=True)
            m = -1 + torch.log(1 + torch.exp(wu))
            index_1, index_2 = torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[0], torch.where((u * w).sum(dim=-1, keepdim=True)<-1)[1]
            u[[index_1], [index_2], :] = u[[index_1], [index_2], :] + (m[[index_1], [index_2], :] - wu[[index_1], [index_2], :]) * w[[index_1], [index_2]] / torch.norm(w[[index_1], [index_2], :], p=2, dim=-1) ** 2

        phi = torch.empty(s.shape)
        abs_detJ = torch.empty(m.shape[:2])
        index_1, index_2 = torch.where(m==0)[0], torch.where(m==0)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 0, :]).sum(dim=-1, keepdim=True) + b[index_1, 0, :])**2 * w[index_1, 0, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 0, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==1)[0], torch.where(m==1)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 1, :]).sum(dim=-1, keepdim=True) + b[index_1, 1, :])**2 * w[index_1, 1, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 1, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==2)[0], torch.where(m==2)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 2, :]).sum(dim=-1, keepdim=True) + b[index_1, 2, :])**2 * w[index_1, 2, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 2, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==3)[0], torch.where(m==3)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 3, :]).sum(dim=-1, keepdim=True) + b[index_1, 3, :])**2 * w[index_1, 3, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 3, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==4)[0], torch.where(m==4)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 4, :]).sum(dim=-1, keepdim=True) + b[index_1, 4, :])**2 * w[index_1, 4, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 4, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==5)[0], torch.where(m==5)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 5, :]).sum(dim=-1, keepdim=True) + b[index_1, 5, :])**2 * w[index_1, 5, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 5, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==6)[0], torch.where(m==6)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 6, :]).sum(dim=-1, keepdim=True) + b[index_1, 6, :])**2 * w[index_1, 6, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 6, :]).sum(dim=-1))

        index_1, index_2 = torch.where(m==7)[0], torch.where(m==7)[1]
        phi[index_1, index_2, :] = 1 - torch.tanh((s[index_1, index_2, :] * w[index_1, 7, :]).sum(dim=-1, keepdim=True) + b[index_1, 7, :])**2 * w[index_1, 7, :]
        abs_detJ[index_1, index_2] = torch.abs(1 + (phi[index_1, index_2, :] * u[index_1, 7, :]).sum(dim=-1))

        return torch.log(1e-4 + abs_detJ)

    # def modified_u(self, w, u): 
    #     wu = w.T @ u
    #     m = -1 + torch.log(1 + torch.exp(wu))

    #     return u + (m - wu) * w / torch.norm(w, p=2, dim=0)**2


class CondPlanarFlows_Measurement(nn.Module): 
    def __init__(self, dim=1, N_m=8, layer=2):
        super().__init__()
        self.layers = [CondPlanarTrans_Measurement(dim, N_m) for _ in range(layer)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, m, s, o): 
        log_detJ = 0. 

        for flow in self.layers: 
            log_detJ += flow.log_detJ(m, s, o)
            s = flow(m, s, o)
        
        return s, log_detJ


class dynamic_NN(nn.Module): 
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


