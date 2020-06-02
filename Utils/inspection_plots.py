import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch

### Scalar variable plots ###
def plot_rewards(rewards):
    r = np.array(rewards).astype(bool)
    spikes = np.arange(len(r))[r]
    flag = True
    for s in spikes:
        if flag:
            plt.axvline(s, alpha=0.7, c='g', label='rewards')
            flag = False
        else:
            plt.axvline(s, alpha=0.7, c='g')
            
def plot_V(d, t_min=0, t_max=-1):
    plt.figure(figsize=(8,6))
    timesteps = np.arange(len(d['values'][t_min:t_max]))
    plt.plot(timesteps, d['values'][t_min:t_max], label='critic prediciton')
    plt.plot(timesteps, d['trg_values'][t_min:t_max], label='critic target')
    plot_rewards(d['rewards'][t_min:t_max])
    plt.legend(fontsize=13)
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('State value', fontsize=16)
    plt.show()
    
def plot_critic_loss(d, t_min=0, t_max=-1):
    plt.figure(figsize=(8,6))
    timesteps = np.arange(len(d['values'][t_min:t_max]))
    plt.plot(timesteps, d['critic_losses'][t_min:t_max])
    plot_rewards(d['rewards'][t_min:t_max])
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Critic loss', fontsize=16)
    plt.show()
    
def plot_actor_loss(d, t_min=0, t_max=-1):
    plt.figure(figsize=(8,6))
    timesteps = np.arange(len(d['values'][t_min:t_max]))
    plt.plot(timesteps, d['advantages'][t_min:t_max], label='estimated advantages')
    plt.plot(timesteps, d['actor_losses'][t_min:t_max], label='actor losses')
    plot_rewards(d['rewards'][t_min:t_max])
    plt.legend(fontsize=13)
    plt.xlabel('Timesteps', fontsize=16)
    plt.ylabel('Actor advantages and losses', fontsize=16)
    plt.show()
    
def plot_update_curves(d, t_min=0, t_max=-1):
    plot_V(d, t_min, t_max)
    plot_critic_loss(d, t_min, t_max)
    plot_actor_loss(d, t_min, t_max)
    
### Trajectory visualization ###

def plot_state(d, t, mask_queue=False):
    PLAYER_COLOR = np.array([200,10,10])
    LAST_POS_COLOR = np.array([100,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    CLICK_COLOUR = np.array([255,255,0])
    
    s = d['state_traj'][t][0]
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
   
    if t>0:
        s_last = d['state_traj'][t-1][0]
        player_y, player_x = s_last[0].nonzero()
        rgb_map[player_y, player_x] = LAST_POS_COLOR
        
    beacon_ys, beacon_xs = s[1].nonzero()
    player_y, player_x = s[0].nonzero()
    point_y, point_x = d['action_sel'][t][0]
    
    rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
    rgb_map[player_y, player_x] = PLAYER_COLOR
    rgb_map[point_y, point_x] = CLICK_COLOUR
        
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])
    
def plot_screen_distr(d, t):
    probs = d['action_distr'][t][0].reshape(16,16)
    plt.imshow(probs, 'plasma')
    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    
def plot_screen_and_decision(d, t, mask_queue=False):
    fig = plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    plot_state(d, t, mask_queue)
    
    plt.subplot(122)
    plot_screen_distr(d, t)
    plt.show()

### Synthetic state ###

def plot_synt_state(s):
    PLAYER_COLOR = np.array([200,10,10])
    BEACON_COLOR = np.array([10,10,100])
    BACKGROUND_COLOR = np.array([10,100,10])
    rgb_map = np.full(s.shape[-2:]+(3,), BACKGROUND_COLOR)
    beacon_ys, beacon_xs = s[1].nonzero()
    player_y, player_x = s[0].nonzero()
    rgb_map[beacon_ys, beacon_xs] = BEACON_COLOR
    rgb_map[player_y, player_x] = PLAYER_COLOR
    plt.imshow(rgb_map)
    plt.xticks([])
    plt.yticks([])
    
def gen_synt_state(agent_pos, beacon_pos, n_channels, res, selected=True):
    if n_channels==6 and res==16:
        return gen_6channels_16res_state(agent_pos, beacon_pos, selected)
    elif n_channels==6 and res==32:
        raise Exception("Resolution of 32 not implemented yet for 6 channels")
    elif n_channels==3 and res==16:
        raise Exception("Resolution of 16 not implemented yet for 3 channels")
    elif n_channels==3 and res==32:
        return gen_3channels_32res_state(agent_pos, beacon_pos, selected)
    else:
        raise Exception("Either the resolution of the number of channels are not supported")
    
def gen_3channels_32res_state(agent_pos, beacon_pos, selected=True):
    
    state = np.zeros((3,res,res))
        
    if res == 32:
        for x in range(beacon_pos[0]-2, beacon_pos[0]+3):
            for y in range(beacon_pos[1]-2, beacon_pos[1]+3):
                cond1 = (x == beacon_pos[0]-2) or (x == beacon_pos[0]+2)
                cond2 = (y == beacon_pos[1]-2) or (y == beacon_pos[1]+2)
                if cond1 and cond2:
                    pass
                else:
                    state[1, y, x] = 1
    else:
        raise Exception("Not implemented for this resolution")
        
    if state[1, agent_pos[1], agent_pos[0]] != 1:
        state[0, agent_pos[1], agent_pos[0]] = 1
        if selected:
            state[2, agent_pos[1], agent_pos[0]] = 1
        
    state = state.astype(int)
    return state

def gen_6channels_16res_state(agent_pos, beacon_pos, selected=True):
    """
    Layers:
    0. player pos
    1. beacon pos
    2. agent selected
    3. visibility map
    4. unit density
    5. unit density anti-aliasing
    """
    
    res = 16
    state = np.zeros((6,res,res)).astype(float)
    for x in range(beacon_pos[0]-1, beacon_pos[0]+2):
        for y in range(beacon_pos[1]-1, beacon_pos[1]+2):
            state[1, y, x] = 1
        
    if state[1, agent_pos[1], agent_pos[0]] != 1:
        state[0, agent_pos[1], agent_pos[0]] = 1
        if selected:
            state[2, agent_pos[1], agent_pos[0]] = 1
                           
    # Visibility map
    for x in range(1,res-1):
        for y in range(1,11):
            state[3,y,x] = 2
    
    # Unit density = sum of beacon and player layers
    state[4] = state[0] + state[1]
    
    #Unit density aa -> crude approximation
    state[5, agent_pos[1], agent_pos[0]] += 4 # agent density all in one cell
    for x in range(beacon_pos[0]-1, beacon_pos[0]+2):
        for y in range(beacon_pos[1]-1, beacon_pos[1]+2):
            cond1 = (x == beacon_pos[0]-1) or (x == beacon_pos[0]+1)
            cond2 = (y == beacon_pos[1]-1) or (y == beacon_pos[1]+1)
            if (x == beacon_pos[0]) and (y == beacon_pos[1]):
                state[5, y, x] += 16
            elif cond1 and cond2:
                state[5, y, x] += 3
            else:
                state[5, y, x] += 12
    return state              

def compute_value_map(agent, beacon_pos, n_channels, res):
    v_map = np.zeros((res,res))
    for x in range(res):
        for y in range(res):
            s = gen_synt_state([x,y], beacon_pos, n_channels, res)
            s = torch.from_numpy(s).float().to(agent.device).unsqueeze(0)
            with torch.no_grad():
                V = agent.AC.V_critic(s).squeeze()
            v_map[y,x] = V.cpu().numpy()
    return v_map

def compute_value_map_6channels_16res(agent, beacon_pos):
    res = 16
    v_map = np.zeros((res,res))
    for x in range(res):
        for y in range(res):
            s = gen_6channels_16res_state([x,y], beacon_pos)
            s = torch.from_numpy(s).float().to(agent.device).unsqueeze(0)
            with torch.no_grad():
                V = agent.AC.V_critic(s).squeeze()
            v_map[y,x] = V.cpu().numpy()
    return v_map

def plot_value_map(agent, beacon_pos, n_channels, res):
    v_map = compute_value_map(agent, beacon_pos, n_channels, res)
    
    fig = plt.figure(figsize=(14,6))
    
    plt.subplot(121)
    plt.title("Beacon position", fontsize = 16)
    plot_synt_state(gen_synt_state(beacon_pos, beacon_pos, n_channels, res))
    
    plt.subplot(122)
    plt.imshow(v_map, cmap='plasma')
    plt.title("Value map", fontsize = 16)
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    return
 
def plot_value_map_at_step(agent, step_idx, PID, beacon_pos, n_channels, res):
    agent.AC.load_state_dict(torch.load("Results/MoveToBeacon/Checkpoints/"+PID+"_"+str(step_idx)))
    agent.AC.to(agent.device) 
    v_map = compute_value_map(agent, beacon_pos, n_channels, res)
    plt.imshow(v_map, cmap='plasma')
    plt.title("Value map - step %d"%step_idx, fontsize = 16)
    plt.xticks([])
    plt.yticks([])
    
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    plt.show()
    return
    
def plot_value_maps(agent, PID, init_step, step_jump, n_jumps, beacon_pos, n_channels, res):
    fig = plt.figure(figsize=(8,6))
    
    plt.title("Beacon position", fontsize = 16)
    state = gen_synt_state(beacon_pos, beacon_pos, n_channels, res)
    plot_synt_state(state)
    
    for n in range(1,n_jumps+1):
        fig = plt.figure(figsize=(8,6))
        
        step_idx = init_step + step_jump*n
        plot_value_map_at_step(agent, step_idx, PID, beacon_pos, n_channels, res)
    return
        
def plot_decision_maps(agent, PID, init_step, step_jump, n_jumps, agent_pos_lst=[[2,2],[11,11],[16,16]], 
                       beacon_pos = [16,16], selected=True, n_channels=6, res=16):
    
    fig = plt.figure(figsize=(14,6))
    
    states = []
    N = len(agent_pos_lst)
    for i in range(N):
        pos = agent_pos_lst[i]
        plt.subplot(1,3,i+1)
        if i == 1:
            plt.title("States considered", fontsize=16)
        s = gen_synt_state(pos, beacon_pos, n_channels, res)
        plot_synt_state(s)
        states.append(s)
    plt.show()
    
    state = torch.from_numpy(np.array(states)).float().to(agent.device)
    
    for n in range(1,n_jumps+1):
        fig = plt.figure(figsize=(14,6))
        
        step_idx = init_step + step_jump*n
        plot_map_at_step(agent, step_idx, PID, state, n_channels, res)
    
def plot_map_at_step(agent, step_idx, PID, state, n_channels, res):
    
    agent.AC.load_state_dict(torch.load("Results/MoveToBeacon/Checkpoints/"+PID+"_"+str(step_idx)))
    agent.AC.to(agent.device) 
    
    with torch.no_grad():
        #print('state.shape: ', state.shape)
        spatial_features = agent.AC.spatial_features_net(state)
        #print("spatial_features.shape: ", spatial_features.shape)
        screen_arg, screen_log_prob, screen_distr = agent.AC.sample_param(spatial_features, 'screen')
    N = screen_distr.shape[0]
    screen_distr = screen_distr.cpu().numpy().reshape(N, res, res)
    M = screen_distr.max()
    m = screen_distr.min()
    
    for i in range(N):
        plt.subplot(1,3,i+1)
        plt.imshow(screen_distr[i], 'plasma', vmax=M, vmin=m)
        plt.xticks([])
        plt.yticks([])
        if i == 1:
            plt.title("Step %d"%step_idx, fontsize = 16)
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
    plt.show()   
    
def print_action_info(d, t):
    print("\nStep %d"%t)
    a_distr = d['action_distr'][t][0]
    select_distr = d['selectall_distr'][t][0]
    queue_distr = d['queue_distr'][t][0]
    point_sel = d['spatial_sel'][t][0]
    adv = d['advantages'][t]
    print("_NO_OP: \t%.2f"%(a_distr[0]*100))
    print("_SELECT_ARMY:  %.2f - _SELECT_ALL: %.2f"%(a_distr[1]*100, select_distr[0]*100))
    print("_MOVE_SCREEN:   %.2f - _NOT_QUEUED: %.2f - POINT: (%d,%d)"%(a_distr[2]*100, queue_distr[0]*100, point_sel[0], point_sel[1]))
    print("Action selected: ", d['action_sel'][t])
    print("Queued: ", d['queue_sel'][t])
    print("Move advantage: %.4f"%adv)
   