import torch
import torch.multiprocessing as mp
import time
import numpy as np
import string
import random
import copy

from Utils.A2C_inspection import *
from Env.test_env import Sandbox 
debug=False

def gen_PID():
    ID = ''.join([random.choice(string.ascii_letters) for _ in range(4)])
    ID = ID.upper()
    return ID


def worker(worker_id, master_end, worker_end, game_params, max_steps):
    master_end.close()  # Forbid worker to use the master end for messaging
    np.random.seed() # sets random seed for the environment
    env = Sandbox(max_steps=max_steps, random_seed=np.random.randint(10000), **game_params)
    
    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            state_trg, reward, done = env.step(data)
            # Always bootstrap when episode finishes (in MoveToBeacon there is no real end)
            if done:
                bootstrap = True
            else:
                bootstrap = False
                
            # ob_trg is the state used as next state for the update
            # ob is the new state used to decide the next action 
            # (different if the episode ends and another one begins)
            if done:
                state = env.reset()
            else:
                state = state_trg
                
            worker_end.send((state, reward, done, bootstrap, state_trg))
            
        elif cmd == 'reset':
            state = env.reset()
            worker_end.send((state))
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes, game_params, max_steps):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end, game_params, max_steps))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        states, rews, dones, bootstraps, trg_states = zip(*results)
        return np.stack(states), np.stack(rews), np.stack(dones), np.stack(bootstraps), np.stack(trg_states)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        states = [master_end.recv() for master_end in self.master_ends]
        return np.stack(states)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True
            
def train_batched_A2C(agent, game_params, lr, n_train_processes, max_train_steps, 
                      unroll_length, max_episode_steps, test_interval=100, num_tests=5):
    
    test_env = Sandbox(max_steps=max_episode_steps, **game_params)
    envs = ParallelEnv(n_train_processes, game_params, max_episode_steps)

    optimizer = torch.optim.Adam(agent.AC.parameters(), lr=lr)
    PID = gen_PID()
    print("Process ID: ", PID)
    score = []
    critic_losses = [] 
    actor_losses = []
    entropy_losses = []
    
    step_idx = 0
    while step_idx < max_train_steps:
        s_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list()
        log_probs = []
        entropies = []
        s = envs.reset()
        for _ in range(unroll_length):

            a, log_prob, entropy = agent.step(s)
            log_probs.append(log_prob)
            entropies.append(entropy)

            s_prime, r, done, bootstrap, s_trg = envs.step(a)
            s_lst.append(s)
            r_lst.append(r)
            done_lst.append(done)
            bootstrap_lst.append(bootstrap)
            s_trg_lst.append(s_trg)

            s = s_prime
            step_idx += 1 #n_train_processes

        s_lst = np.array(s_lst).transpose(1,0,2,3,4)
        r_lst = np.array(r_lst).transpose(1,0)
        done_lst = np.array(done_lst).transpose(1,0)
        bootstrap_lst = np.array(bootstrap_lst).transpose(1,0)
        s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4)

        critic_loss, actor_loss, entropy_term = agent.compute_ac_loss(r_lst, log_probs, entropies, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        
        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropy_losses.append(entropy_term.item())
        
        
        ### Test time ###
        if step_idx % test_interval == 0:
            avg_score, inspector = test(step_idx, agent, test_env, PID, num_tests)
            score.append(avg_score)
            # save episode for inspection and model weights at that point
            inspector.save_dict()
            torch.save(agent.AC.state_dict(), "Results/Checkpoints/"+PID+"_"+str(step_idx))
    envs.close()
    
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropy_losses)
    return score, losses, agent, PID


def train_from_checkpoint(agent, game_params, lr, n_train_processes, max_train_steps, 
                      unroll_length, max_episode_steps, PID, step_idx, test_interval=100, num_tests=5):
    
    agent.AC.load_state_dict(torch.load("Results/Checkpoints/"+PID+"_"+str(step_idx)))
    agent.AC.to(agent.device) 
    
    test_env = Sandbox(max_steps=max_episode_steps, **game_params)
    envs = ParallelEnv(n_train_processes, game_params, max_episode_steps)

    optimizer = torch.optim.Adam(agent.AC.parameters(), lr=lr)
    
    score = []
    critic_losses = [] 
    actor_losses = []
    entropy_losses = []
    
    while step_idx < max_train_steps:
        s_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list()
        log_probs = []
        entropies = []
        s = envs.reset()
        for _ in range(unroll_length):

            a, log_prob, entropy = agent.step(s)
            log_probs.append(log_prob)
            entropies.append(entropy)

            s_prime, r, done, bootstrap, s_trg = envs.step(a)
            s_lst.append(s)
            r_lst.append(r)
            done_lst.append(done)
            bootstrap_lst.append(bootstrap)
            s_trg_lst.append(s_trg)

            s = s_prime
            step_idx += 1 #n_train_processes

        s_lst = np.array(s_lst).transpose(1,0,2,3,4)
        r_lst = np.array(r_lst).transpose(1,0)
        done_lst = np.array(done_lst).transpose(1,0)
        bootstrap_lst = np.array(bootstrap_lst).transpose(1,0)
        s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4)

        critic_loss, actor_loss, entropy_term = agent.compute_ac_loss(r_lst, log_probs, entropies, 
                                                                 s_lst, done_lst, bootstrap_lst, s_trg_lst)

        
        loss = (critic_loss + actor_loss).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        entropy_losses.append(entropy_term.item())
        
        ### Test time ###
        if step_idx % test_interval == 0:
            avg_score, inspector = test(step_idx, agent, test_env, PID, num_tests)
            score.append(avg_score)
            # save episode for inspection and model weights at that point
            inspector.save_dict()
            torch.save(agent.AC.state_dict(), "Results/Checkpoints/"+PID+"_"+str(step_idx))
    envs.close()
    
    losses = dict(critic_losses=critic_losses, actor_losses=actor_losses, entropies=entropy_losses)
    return score, losses, agent, PID

def test(step_idx, agent, test_env, process_ID, num_test=5):
    score = 0.0
    done = False
    
    ### Standard tests ###
    for _ in range(num_test-1):
        
        s = test_env.reset()
        s = s[np.newaxis, ...] # add batch dim
        
        while not done:
            a, log_prob, probs = agent.step(s)
            s_prime, reward, done = test_env.step(a[0])
            s_prime = s_prime[np.newaxis, ...] # add batch dim
            s = s_prime
            score += reward
        done = False
        
    ### Inspection test ###
    G, inspector = inspection_test(step_idx, agent, test_env, process_ID)
    score += G
    print(f"Step # : {step_idx}, avg score : {score/num_test:.1f}")
    return score/num_test, inspector

def inspection_test(step_idx, agent, test_env, process_ID):
    inspector = InspectionDict(step_idx, process_ID)
    
    s = test_env.reset()
    s = s[np.newaxis, ...] # add batch dim
    
    done = False
    G = 0.0
    # list used for update
    s_lst, r_lst, done_lst, bootstrap_lst, s_trg_lst = list(), list(), list(), list(), list()
    log_probs = []
    entropies = []
    while not done:
        inspector.dict['state_traj'].append(s)
        a, log_prob, entropy, step_dict = inspection_step(agent, s)
        inspector.store_step(step_dict)
        log_probs.append(log_prob)
        entropies.append(entropy)
        
        s_prime, reward, done = test_env.step(a[0])
        s_prime = s_prime[np.newaxis, ...] # add batch dim
       
        if done:
            bootstrap = True
        else:
            bootstrap = False
            
        s_lst.append(s)
        r_lst.append(reward)
        done_lst.append(done)
        bootstrap_lst.append(bootstrap)
        s_trg_lst.append(s_prime)
            
        s = s_prime
        G += reward
        
    inspector.dict['rewards'] = r_lst
    s_lst = np.array(s_lst).transpose(1,0,2,3,4)
    r_lst = np.array(r_lst).reshape(1,-1)
    done_lst = np.array(done_lst).reshape(1,-1)
    bootstrap_lst = np.array(bootstrap_lst).reshape(1,-1)
    s_trg_lst = np.array(s_trg_lst).transpose(1,0,2,3,4)    
    update_dict = inspection_update(agent, r_lst, log_probs, entropies, s_lst, 
                                    done_lst, bootstrap_lst, s_trg_lst)
    inspector.store_update(update_dict)
    return G, inspector