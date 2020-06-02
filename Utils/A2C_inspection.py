import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F 

class InspectionDict():
    def __init__(self, step_idx, PID):
        self.step_idx = step_idx
        self.PID = PID
        self.dict = dict(
                        state_traj = [],
                        rewards = [],
                        action_distr = [],
                        action_sel = [],
                        values = None,
                        trg_values = None,
                        critic_losses = None,
                        advantages = None,
                        actor_losses = None)
        
    def store_step(self, step_dict):
        # store every other trajectory variable except state_traj
        for k in step_dict:
            self.dict[k].append(step_dict[k])
        return
    
    def store_update(self, update_dict):
        for k in update_dict:
            self.dict[k] = update_dict[k]
        return
    
    def save_dict(self, path='Results/Inspection/'):
        np.save(path+self.PID+"_"+str(self.step_idx), self.dict)
        return
    
def inspection_step(agent, state):
    state = torch.from_numpy(state).float().to(agent.device)
    with torch.no_grad():
        action, log_prob, probs = agent.AC.pi(state)
        entropy = agent.compute_entropy(probs)
    step_dict = {}
    p = probs.detach().cpu().numpy() 
    step_dict['action_distr'] = p
    step_dict['action_sel'] = action

    return action, log_prob, torch.mean(entropy), step_dict

def inspection_update(agent, rewards, log_probs, entropies, states, done, bootstrap, trg_states): 
    old_states = torch.tensor(states).float().to(agent.device).reshape((-1,)+states.shape[2:])
    V_trg = []
    for n in range(1, agent.n_steps + 1):
        n_step_V_trg = agent.compute_n_step_V_trg(n, rewards, done, bootstrap, states)
        V_trg.append(n_step_V_trg)
    V_trg = torch.mean(torch.stack(V_trg, axis=0), axis=0)
    log_probs = torch.stack(log_probs).to(agent.device).transpose(1,0).reshape(-1)
    entropies = torch.stack(entropies, axis=0).to(agent.device).reshape(-1)

    values, trg_values, critic_losses = inspect_critic_loss(agent, old_states, V_trg)

    advantages, actor_losses = inspect_actor_loss(agent, log_probs, entropies, old_states, V_trg)

    update_dict = dict(values=values, 
                       trg_values=trg_values, 
                       critic_losses=critic_losses, 
                       advantages=advantages, 
                       actor_losses=actor_losses )
    return update_dict

def inspect_critic_loss(agent, old_states, V_trg):
    with torch.no_grad():
        V = agent.AC.V_critic(old_states).squeeze()
        V = V.cpu().numpy() 
        V_trg = V_trg.cpu().numpy()
        critic_losses = (V-V_trg)**2
    return V, V_trg, critic_losses

def inspect_actor_loss(agent, log_probs, entropies, old_states, V_trg):
    with torch.no_grad():
        V_pred = agent.AC.V_critic(old_states).squeeze()
        A = V_trg - V_pred
        #A = (A - A.mean())/(A.std()+1e-5)
        policy_gradient = - log_probs*A
    A = A.cpu().numpy()
    pg = policy_gradient.cpu().numpy()
    return A, pg
