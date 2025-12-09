import sys, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable, grad
from torch.func import vmap
import torch.autograd as autograd
from utils import nvp
from utils import SK
from utils import PT

cuda = True if torch.cuda.is_available() else False
device = 'cpu' if not cuda else 'cuda'

dim = 256
n_disorder = 1
T_list = np.asarray(list(np.linspace(0,1,11)[1:]) + list(np.linspace(1,5,11))[1:])
NT = len(T_list)
n_batches = 2000 + 1
batch_size = 50
AdamLR = 1e-4

path = 'data/REVERSETEST' + str(dim)+'/'
if not os.path.exists(path):
    os.makedirs(path)

log_filename = path + f'training_dim_{dim}.log'
log_file = open(log_filename, 'w')

prior = distributions.MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device))

class Interpolant:
    def alpha(self, t):
        return 1.0 - t
    
    def dotalpha(self, t):
        return -1.0 + 0*t
    
    def beta(self, t):
        return t
    
    def dotbeta(self, t):
        return 1.0 + 0*t
    
    def _single_xt(self, x0, x1, t):
        return self.alpha(t)*x0 + self.beta(t)*x1
    
    def _single_dtxt(self, x0, x1, t):
        return self.dotalpha(t)*x0 + self.dotbeta(t)*x1
    
    def xt(self, x0, x1, t):
        return vmap(self._single_xt, in_dims=(0, 0, 0))(x0,x1,t)
    
    def dtxt(self, x0, x1, t):
        return vmap(self._single_dtxt, in_dims=(0, 0, 0))(x0,x1,t)
    
interpolant = Interpolant()

class VelocityField(torch.nn.Module):
    def __init__(self, d,  hidden_sizes = [256, 256], activation=torch.nn.ReLU):
        super(VelocityField, self).__init__()
        
        layers = []
        prev_dim = d + 1  #
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_size))
            layers.append(activation())
            prev_dim = hidden_size  

        layers.append(torch.nn.Linear(prev_dim, d))
        
        self.net = torch.nn.Sequential(*layers)
    
    def _single_forward(self, x, t):  
        t = t.unsqueeze(-1)
        return self.net(torch.cat((x, t)))
    
    def forward(self, x, t):
        return vmap(self._single_forward, in_dims=(0,0), out_dims=(0))(x,t)

b =  VelocityField(dim, hidden_sizes=[1024, 1024, 1024])

def log_pi(x, Sigma_inv_torch):
    term1 = -0.5*torch.diagonal(torch.mm(torch.mm(x, Sigma_inv_torch), torch.transpose(x, 1, 0)))
    y = torch.transpose(torch.abs(torch.mm(Sigma_inv_torch, torch.transpose(x, 1, 0))), 1, 0)
    term2 = torch.sum(y + torch.log(1 + torch.exp(-2*y)), 1)
    return term1 + term2 

def single_step_flow(z, b):
    z = z.clone().requires_grad_(True)       # shape: (batch_size, dim)
    batch_size, d = z.shape
    t_mid = torch.full((batch_size,), 0.5, device=z.device)  # pick a mid time t=0.5 for approximation
    v = b(z, t_mid)                          # velocity field at z, t=0.5
    x = z + v                                # single-step from t=0->1

    div = torch.zeros(batch_size, device=z.device)
    for i in range(d):
        grad_i = autograd.grad(v[:, i].sum(), z, create_graph=True)[0]  # shape: (batch_size, dim)
        div += grad_i[:, i]
    log_det_J = div  
    return x, log_det_J

def quick_log_prob_of_z(z, flow, prior):
    x, log_det_J = single_step_flow(z, flow)
    log_pz = prior.log_prob(z)
    return log_pz + log_det_J

def quick_log_prob_of_z_minus(z, flow, prior):
    x, log_det_J = single_step_flow(-z, flow)
    log_pz = prior.log_prob(z)
    return log_pz + log_det_J

def quick_log_prob_of_z_sym(z, flow, prior):
    log_prob = quick_log_prob_of_z(z, flow, prior)            # log_prob_of_z
    log_prob_minus = quick_log_prob_of_z_minus(z, flow, prior) # log_prob_of_z_minus
    return log_prob + torch.log(0.5 * (1 + torch.exp(log_prob_minus - log_prob)))

def single_step_loss(z, flow, Sigma_inv_torch, prior):
    x, _ = single_step_flow(z, flow)
    log_prob_sym = quick_log_prob_of_z_sym(z, flow, prior)
    log_pi_term = log_pi(x, Sigma_inv_torch)

    return (log_prob_sym - log_pi_term).mean()


for k in range(0, n_disorder):
        
    for iT in range(len(T_list)):

        T = T_list[iT]
        log_line = f'\nT = {T:.3f}, seed = {k}'
        print(log_line)
        log_file.write(log_line + '\n')
        log_file.flush()

        J = SK.generate_J(dim, seed=k)

        beta = 1/T
        Jbar = beta*J
        Jbar_eigs = np.linalg.eig(Jbar)[0]
        bareps = beta*1e-2
        Dbar = (max(0,-min(Jbar_eigs)) + bareps)*np.eye(dim)
        Sigma_inv = Jbar + Dbar
        Sigma = np.linalg.inv(Sigma_inv)
        Sigma_torch = torch.from_numpy(Sigma).type(torch.FloatTensor).to(device)
        Sigma_inv_torch = torch.from_numpy(Sigma_inv).type(torch.FloatTensor).to(device)  

        flow = b
        if cuda: flow = flow.cuda()

        torch.save(flow, path + 'b' + str(dim) + '_iT_' + str(iT) + '_batch_' + str(0) + '_disorder_' + str(k))

        optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=AdamLR)

        loss_history = []
        iters = []

        for n in range(n_batches):

            z = prior.sample((batch_size, 1)).reshape((batch_size, dim))
            loss = single_step_loss(z, flow, Sigma_inv_torch, prior)      
 
            loss_history.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            log_line = f'finished epoch {n}, loss {loss.item():.3f}, avg. loss = {np.mean(loss_history):.3f}'
            print(log_line)
            log_file.write(log_line + '\n')
            log_file.flush()

        np.save(path + 'T_list', T_list)
        np.save(path + 'loss_dim_' + str(dim) + '_iT_' + str(iT) + '_disorder_' + str(k), loss_history)
        torch.save(flow, path + 'b' + str(dim) + '_iT_' + str(iT) + '_batch_' + str(n+1) + '_disorder_' + str(k))

log_file.close()





