import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import Categorical

from AC_modules.ActorCriticArchitecture import *

debug = False

class SpatialA2C():
    """
    Advantage Actor-Critic RL agent for BoxWorld environment described in the paper
    Relational Deep Reinforcement Learning.
    
    Notes
    -----
      
    """ 
    
    def __init__(self,  gamma, H=1e-3, n_steps=20, device='cpu', shared=False, 
                 actor_model={}, critic_model={}, actor_dict={}, critic_dict={},
                 spatial_model={}, spatial_dict={}, shared_act_dict={}, shared_crit_dict={}):
        self.gamma = gamma
        self.n_steps = n_steps
        self.H = H
        if shared:
            self.AC = SharedSpatialActorCritic(spatial_model, spatial_dict, shared_act_dict, shared_crit_dict)
        else:
            self.AC = SpatialActorCritic(actor_model, critic_model, actor_dict, critic_dict)
        self.device = device 
        self.AC.to(self.device) 

    def step(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        action, log_prob, probs = self.AC.pi(state)
        entropy = self.compute_entropy(probs)
        return action, log_prob, torch.mean(entropy)

    def compute_entropy(self, probs):
        """
        Computes negative entropy of a batch (b, n_actions) of probabilities.
        Returns the entropy of each sample in the batch (b,)
        """
        # add a small regularization to probs
        probs = probs + torch.tensor([1e-5]).float().to(self.device)
        entropy = torch.sum(probs*torch.log(probs), axis=1)
        return entropy
    
    def compute_ac_loss(self, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
        # merge batch and episode dimensions
        old_states = torch.tensor(states).float().to(self.device).reshape((-1,)+states.shape[2:])

        ### Use as V target the mean of 1-step to n-step V targets
        V_trg = []
        average_n_steps = False
        if average_n_steps:
            for n in range(1, self.n_steps + 1):
                n_step_V_trg = self.compute_n_step_V_trg(n, rewards, done, bootstrap, states)
                V_trg.append(n_step_V_trg)
            V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
        else:
            V_trg = self.compute_n_step_V_trg(self.n_steps, rewards, done, bootstrap, states)
        if debug: print("V_trg.shape: ", V_trg.shape)
       
        ### Wrap variables into tensors - merge batch and episode dimensions ###    
        log_probs = torch.stack(log_probs).to(self.device).transpose(1,0).reshape(-1)
        if debug: print("log_probs.shape: ", log_probs.shape)
        entropies = torch.stack(entropies, axis=0).to(self.device).reshape(-1)
        if debug: print("entropies.shape: ", entropies.shape)
        
        ### Update critic and then actor ###
        critic_loss = self.compute_critic_loss(old_states, V_trg)
        actor_loss, entropy = self.compute_actor_loss(log_probs, entropies, old_states, V_trg)

        return critic_loss, actor_loss, entropy
    
    def compute_n_step_V_trg(self, n_steps, rewards, done, bootstrap, states):
        n_step_rewards, episode_mask, n_steps_mask_b = self.compute_n_step_rewards(rewards, done, n_steps)
        done[bootstrap] = False 
        new_states, Gamma_V, done = self.compute_n_step_states(states, done, episode_mask, n_steps_mask_b)
        
        new_states = torch.tensor(new_states).float().to(self.device).reshape((-1,)+states.shape[2:])
        done = torch.LongTensor(done.astype(int)).to(self.device).reshape(-1)
        n_step_rewards = torch.tensor(n_step_rewards).float().to(self.device).reshape(-1)
        Gamma_V = torch.tensor(Gamma_V).float().to(self.device).reshape(-1)
        
        with torch.no_grad():
            V_pred = self.AC.V_critic(new_states).squeeze()
            V_trg = (1-done)*Gamma_V*V_pred + n_step_rewards
            V_trg = V_trg.squeeze()
        if debug: print("V_trg.shape; ", V_trg.shape)
        return V_trg
    
    def compute_critic_loss(self, old_states, V_trg):
        V = self.AC.V_critic(old_states).squeeze()
        loss = F.mse_loss(V, V_trg)
        return loss
    
    def compute_actor_loss(self, log_probs, entropies, old_states, V_trg):
        with torch.no_grad():
            V_pred = self.AC.V_critic(old_states).squeeze()
        A = V_trg - V_pred
        #A = (A - A.mean())/(A.std()+1e-5)
        policy_gradient = - log_probs*A
        if debug:
            print("V_trg.shape: ",V_trg.shape)
            print("V_trg: ", V_trg)
            print("V_pred.shape: ",V_pred.shape)
            print("V_pred: ", V_pred)
            print("A.shape: ", A.shape)
            print("A: ", A)
            print("policy_gradient.shape: ", policy_gradient.shape)
            print("policy_gradient: ", policy_gradient)
        policy_grad = torch.mean(policy_gradient)
        if debug: print("policy_grad: ", policy_grad)
        
        entropy = torch.mean(entropies)
        loss = policy_grad + self.H*entropy
        if debug: print("Actor loss: ", loss)
             
        return loss, entropy
                
    def compute_n_step_rewards(self, rewards, done, n_steps=None):
        """
        Computes n-steps discounted reward padding with zeros the last elements of the trajectory.
        This means that the rewards considered are AT MOST n, but can be less for the last n-1 elements.
        """
        if n_steps is None:
            n_steps = self.n_steps
            
        B = done.shape[0]
        T = done.shape[1]
        if debug:
            print("batch size: ", B)
            print("unroll len: ", T)
        
        
        # Compute episode mask (i-th row contains 1 if col j is in the same episode of col i, 0 otherwise)
        episode_mask = [[] for _ in range(B)]
        last = [-1 for _ in range(B)]
        xs, ys = np.nonzero(done)
        # Add done at the end of every batch to avoid exceptions -> not used in real target computations
        xs = np.concatenate([xs, np.arange(B)])
        ys = np.concatenate([ys, np.full(B, T-1)])
        for x, y in zip(xs, ys):
            m = [1 if (i > last[x] and i <= y) else 0 for i in range(T)]
            for _ in range(y-last[x]):
                episode_mask[x].append(m)
            last[x] = y
        episode_mask = np.array(episode_mask)
        
        # Compute n-steps mask and repeat it B times
        n_steps_mask = []
        for i in range(T):
            m = [1 if (j>=i and j<i+n_steps) else 0 for j in range(T)]
            n_steps_mask.append(m)
        n_steps_mask = np.array(n_steps_mask)
        n_steps_mask_b = np.repeat(n_steps_mask[np.newaxis,...] , B, axis=0)
        
        # Broadcast rewards to use multiplicative masks
        rewards_repeated = np.repeat(rewards[:,np.newaxis,:], T, axis=1)
        
        # Exponential discount factor
        Gamma = np.array([self.gamma**i for i in range(T)]).reshape(1,-1)
        if debug:
            print("Gamma.shape: ", Gamma.shape)
            print("rewards_repeated.shape: ", rewards_repeated.shape)
            print("episode_mask.shape: ", episode_mask.shape)
            print("n_steps_mask_b.shape: ", n_steps_mask_b.shape)
        n_steps_r = (Gamma*rewards_repeated*episode_mask*n_steps_mask_b).sum(axis=2)/Gamma
        return n_steps_r, episode_mask, n_steps_mask_b
    
    def compute_n_step_states(self, trg_states, done, episode_mask, n_steps_mask_b):
        """
        Computes n-steps target states (to be used by the critic as target values together with the
        n-steps discounted reward). For last n-1 elements the target state is the last one available.
        Adjusts also the `done` mask used for disabling the bootstrapping in the case of terminal states
        and returns Gamma_V, that are the discount factors for the target state-values, since they are 
        n-steps away (except for the last n-1 states, whose discount is adjusted accordingly).
        
        Return
        ------
        new_states, Gamma_V, done: arrays with first dimension = len(states)-1
        """
        
        B = done.shape[0]
        T = done.shape[1]
        V_mask = episode_mask*n_steps_mask_b
        b, x, y = np.nonzero(V_mask)
        V_trg_index = [[] for _ in range(B)]
        for b_i in range(B):
            valid_x = (b==b_i)
            for i in range(T):
                matching_x = (x==i)
                V_trg_index[b_i].append(y[valid_x*matching_x][-1])
        V_trg_index = np.array(V_trg_index)
        
        cols = np.array([], dtype=np.int)
        rows = np.array([], dtype=np.int)
        for i, v in enumerate(V_trg_index):
            cols = np.concatenate([cols, v], axis=0)
            row = np.full(V_trg_index.shape[1], i)
            rows = np.concatenate([rows, row], axis=0)
        new_states = trg_states[rows, cols].reshape(trg_states.shape)
        pw = V_trg_index - np.arange(V_trg_index.shape[1]) + 1
        Gamma_V = self.gamma**pw
        shifted_done = done[rows, cols].reshape(done.shape)
        return new_states, Gamma_V, shifted_done