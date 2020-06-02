import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import itertools as it

from AC_modules.Networks import *

debug = False

class SpatialActorCritic(nn.Module):
    """
    Actor samples point on a screen; critic extract state-value.
    """
    def __init__(self, actor_model, critic_model, actor_dict, critic_dict):
        super(SpatialActorCritic, self).__init__()

        # Networks
        self.actor = actor_model(**actor_dict)
        self.critic = critic_model(**critic_dict) 

    def pi(self, state):
        action, log_prob, probs = self.actor(state)
        return action, log_prob, probs
    
    def V_critic(self, state):
        V = self.critic(state)
        return V
    
class SharedSpatialActorCritic(nn.Module):
    """
    Actor samples point on a screen; critic extract state-value.
    """
    def __init__(self, spatial_model, spatial_dict, shared_act_dict, shared_crit_dict):
        super(SharedSpatialActorCritic, self).__init__()

        # Networks
        self.shared_net = spatial_model(**spatial_dict)
        self.actor = SharedActor(**shared_act_dict)
        self.critic = SharedCritic(**shared_crit_dict)
        
        
    def pi(self, state):
        spatial_repr = self.shared_net(state)
        action, log_prob, probs = self.actor(spatial_repr)
        return action, log_prob, probs
    
    def V_critic(self, state):
        spatial_repr = self.shared_net(state)
        V = self.critic(spatial_repr)
        return V
    
class SharedActor(nn.Module):
    def __init__(self, n_channels, linear_size):
        super(SharedActor, self).__init__()
        self.net = SharedActorNet(n_channels, linear_size)
        
    def forward(self, spatial_repr):
        return self.net(spatial_repr)

class SharedCritic(nn.Module):
    def __init__(self, n_channels, linear_size):
        super(SharedCritic, self).__init__()
        self.net = SharedCriticNet(n_channels, linear_size)
  
    def forward(self, spatial_repr):
        return self.net(spatial_repr)