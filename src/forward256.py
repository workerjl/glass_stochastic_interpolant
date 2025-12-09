import sys, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable, grad
from torch.func import vmap
from utils import nvp
from utils import SK
from utils import PT

cuda = True if torch.cuda.is_available() else False
device = 'cpu' if not cuda else 'cuda'

dim = 256
n_disorder = 1
T_list = np.asarray(list(np.linspace(0,1,11)[1:]) + list(np.linspace(1,5,11))[1:])
NT = len(T_list)
n_epochs = 125
batch_size = 50
AdamLR = 1e-4

path = 'data/FORWARDTEST' + str(dim)+'/'
if not os.path.exists(path):
    os.makedirs(path)

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

def _single_loss(b, interpolant, x0, x1, t):
    """
    Interpolant loss function for a single datapoint of (x0, x1, t).
    """
    It   = interpolant._single_xt(  x0, x1, t)
    dtIt = interpolant._single_dtxt(x0, x1, t)
    
    bt          = b._single_forward(It, t)
    loss        = 0.5*torch.sum(bt**2) - torch.sum((dtIt) * bt)
    return loss

loss_fn = vmap(_single_loss, in_dims=(None, None, 0, 0, 0), out_dims=(0), randomness='different')

for k in range(0, n_disorder):
        
    for iT in range(len(T_list)):

        T = T_list[iT]
        print('\nT = %.3f, seed = %i' % (T,k))

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

        fileName = "data/PT_MCMC/xlistA_iT_" + str(iT) + "k_" + str(k) + ".npy"
        if not os.path.isfile(fileName):
            print('converting from s to x:')
            SlistA = np.load("data/PT_MCMC/SlistA_k" + str(k) + ".npy")[:,iT,:]
            xlistA = np.zeros(SlistA.shape)
            for i in range(SlistA.shape[0]):
                xlistA[i,:] = np.random.multivariate_normal(SlistA[i], Sigma)
            np.save(fileName, xlistA)

        else:
            print('loading x data')
            xlistA = np.load(fileName)    

        n_batches = xlistA.shape[0]//batch_size

        flow = b
        if cuda: flow = flow.cuda()

        torch.save(flow, path + 'b' + str(dim) + '_iT_' + str(iT) + '_epoch_' + str(0) + '_disorder_' + str(k))

        optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=AdamLR)

        loss_history = []
        iters = []

        for epoch in range(n_epochs):

            for batch in range(n_batches):

                x_batch = xlistA[batch*batch_size:(batch+1)*batch_size,:]
                x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor).to(device)
                ts  = torch.rand(batch_size)
                x0s = prior.sample((batch_size,))

                loss = loss_fn(flow, interpolant, x0s, x_batch, ts).mean()       
                loss_history.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print('finished epoch %i/%i, loss %.3f' % (epoch, n_epochs, np.mean(loss_history)))

        np.save(path + 'T_list', T_list)
        np.save(path + 'loss_dim_' + str(dim) + '_iT_' + str(iT) + '_disorder_' + str(k), loss_history)
        torch.save(flow, path + 'b' + str(dim) + '_iT_' + str(iT) + '_epoch_' + str(epoch+1) + '_disorder_' + str(k))
