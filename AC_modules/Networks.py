from AC_modules.Layers import *

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

debug = False
    
class CriticNet(nn.Module):
    def __init__(self, in_channels, linear_size, hidden_dim=256):
        super(CriticNet, self).__init__()
        self.flatten_size = 64*((linear_size-2)//2)**2
        self.spatial_net = nn.Sequential(
                        nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, kernel_size=3, stride=2),
                        nn.ReLU()
                        )
        self.critic_net = nn.Sequential(
                        nn.Linear(self.flatten_size, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                        )
    def forward(self, x):
        B = x.shape[0]
        x = self.spatial_net(x)
        x = x.reshape((B,-1))
        V = self.critic_net(x)
        return V
        
class SpatialNet(nn.Module):
    
    def __init__(self, in_channels, linear_size):
        super(SpatialNet, self).__init__()
        
        self.size = linear_size
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
                        )
    
    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        if debug: print("x.shape: ", x.shape)
        x = x.reshape((x.shape[0],-1))
        if debug: print("x.shape: ", x.shape)
        log_probs = F.log_softmax(x, dim=(-1))
        probs = torch.exp(log_probs)
        if debug: 
            print("log_probs.shape: ", log_probs.shape)
            print("log_probs.shape (reshaped): ", log_probs.view(-1, self.size, self.size).shape)
            
        # assume squared space
        x_lin = torch.arange(self.size).unsqueeze(0)
        xx = x_lin.repeat(B,self.size,1)
        if debug: print("xx.shape: ", xx.shape)
        args = torch.cat([xx.permute(0,2,1).view(-1,self.size,self.size,1), xx.view(-1,self.size,self.size,1)], axis=3)
        if debug: print("args.shape (before reshaping): ", args.shape)
        args = args.reshape(B,-1,2)
        if debug: print("args.shape (after reshaping): ", args.shape)
        
        index = Categorical(probs).sample()
        arg = args[torch.arange(B), index].detach().numpy() # and this are the sampled coordinates
        arg_lst = [list(a)  for a in arg]
        log_probs = log_probs.reshape(B, self.size, self.size)
        return arg_lst, log_probs[torch.arange(B), arg[:,0], arg[:,1]], probs  
    
### For shared architectures ###

class SharedNet(nn.Module):
    def __init__(self, in_channels, n_channels):
        super(SharedNet, self).__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(16, n_channels, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        
    def forward(self, x):
        return self.net(x)
        
        
class SharedCriticNet(nn.Module):
    def __init__(self, n_channels, linear_size, hidden_dim=256):
        super(SharedCriticNet, self).__init__()
        self.flatten_size = 64*((linear_size-2)//2)**2
        self.spatial_net = nn.Sequential(
                        nn.Conv2d(n_channels, 64, kernel_size=3, stride=2),
                        nn.ReLU()
                        )
        self.critic_net = nn.Sequential(
                        nn.Linear(self.flatten_size, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1)
                        )
    def forward(self, x):
        B = x.shape[0]
        x = self.spatial_net(x)
        x = x.reshape((B,-1))
        V = self.critic_net(x)
        return V
    
class SharedActorNet(nn.Module):
    
    def __init__(self, n_channels, linear_size):
        super(SharedActorNet, self).__init__()
        
        self.size = linear_size
        self.conv = nn.Conv2d(n_channels, 1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        if debug: print("x.shape: ", x.shape)
        x = x.reshape((x.shape[0],-1))
        if debug: print("x.shape: ", x.shape)
        log_probs = F.log_softmax(x, dim=(-1))
        probs = torch.exp(log_probs)
        if debug: 
            print("log_probs.shape: ", log_probs.shape)
            print("log_probs.shape (reshaped): ", log_probs.view(-1, self.size, self.size).shape)
            
        # assume squared space
        x_lin = torch.arange(self.size).unsqueeze(0)
        xx = x_lin.repeat(B,self.size,1)
        if debug: print("xx.shape: ", xx.shape)
        args = torch.cat([xx.permute(0,2,1).view(-1,self.size,self.size,1), xx.view(-1,self.size,self.size,1)], axis=3)
        if debug: print("args.shape (before reshaping): ", args.shape)
        args = args.reshape(B,-1,2)
        if debug: print("args.shape (after reshaping): ", args.shape)
        
        index = Categorical(probs).sample()
        arg = args[torch.arange(B), index].detach().numpy() # and this are the sampled coordinates
        arg_lst = [list(a)  for a in arg]
        log_probs = log_probs.reshape(B, self.size, self.size)
        return arg_lst, log_probs[torch.arange(B), arg[:,0], arg[:,1]], probs  
   
