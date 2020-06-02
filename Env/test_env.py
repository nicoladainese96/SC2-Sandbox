import numpy as np
import time
import copy

BACKGROUND_COLOR = 0
AGENT_COLOR = 1
GOAL_COLOR = 2
WALL_COLOR = 3

debug = False

class Sandbox():
    
    def __init__(self, res, max_steps=100, n_channels=2, random_seed=None):
        np.random.seed(random_seed)
        self.n_channels = n_channels
        self.res = res
        self.boundary = np.asarray([res, res]) # try to remove this
        self.max_steps = max_steps
        self.current_steps = 0
        self.moves_dict = {'left':[0,-1],
                           'right':[0,1],
                           'down':[1,0],
                           'up':[-1,0]}
        self._init_map()
        
    def _init_map(self):
        if self.n_channels==2 and self.res==16:
            self._init_2chan_16res_map()
        else:
            raise Exception("Not implemented yet")
            
    def _init_2chan_16res_map(self):
        self.state = np.zeros((2,16,16)).astype(int)
        # Init positions
        for i in range(100):
            beacon_pos = np.random.choice(np.arange(1, self.res-1), size=2, replace=True)
            agent_pos = np.random.choice(np.arange(self.res), size=2, replace=True)
            if self._valid_start(agent_pos, beacon_pos):
                break
            if i==99: 
                raise Exception("Init failed")
                
        self.state[0, agent_pos[0], agent_pos[1]] = 1
        self._create_beacon_layer(beacon_pos)

    def _valid_start(self, agent_pos, beacon_pos):
        # check that agent is not inside a 3x3 box centered on the beacon
        cond1 = (agent_pos[0] < beacon_pos[0] - 1) or (agent_pos[0] > beacon_pos[0] + 1) 
        cond2 = (agent_pos[1] < beacon_pos[1] - 1) or (agent_pos[1] > beacon_pos[1] + 1) 
        if cond1 or cond2:
            return True
        else:
            return False
        
    def _create_beacon_layer(self, beacon_pos):
        for x in range(beacon_pos[0]-1, beacon_pos[0]+2):
            for y in range(beacon_pos[1]-1, beacon_pos[1]+2):
                self.state[1, x, y] = 1
    
    def reset(self):
        self._init_map()
        self.current_steps = 0
        return copy.deepcopy(self.state)
    
    def _get_agent_pos(self):
        x, y = np.nonzero(self.state[0])
        return np.array([x[0], y[0]])
    
    def _get_beacon_center(self):
        x, y = np.nonzero(self.state[1])
        return np.array([int(np.mean(x)), int(np.mean(y))])
    
    def _update_agent_pos(self, new_pos):
        self.state[0] = 0
        self.state[0, new_pos[0], new_pos[1]] = 1
        return
    
    def _spawn_beacon(self):
        self.state[1] = 0
        for i in range(100):
            beacon_pos = np.random.choice(np.arange(1, self.res-1), size=2, replace=True)
            agent_pos = self._get_agent_pos()
            if self._valid_start(agent_pos, beacon_pos):
                break
            if i==99: 
                raise Exception("Init failed")
        self._create_beacon_layer(beacon_pos)
    
    def _valid_action(self, action):
        cond1 = action[0] >= 0 and action[0] < self.res
        cond2 = action[1] >= 0 and action[1] < self.res
        if cond1 and cond2:
            return True
        else:
            return False
        
    def _action_to_movement(self, action):
        if not self._valid_action(action):
            raise Exception("Action not valid")
            
        movements = []
        agent_pos = self._get_agent_pos()
        if action[0] > agent_pos[0]:
            movements.append('down')
        elif action[0] < agent_pos[0]:
            movements.append('up')
        else:
            pass
        
        if action[1] > agent_pos[1]:
            movements.append('right')
        elif action[1] < agent_pos[1]:
            movements.append('left')
        else:
            pass
        
        if len(movements) != 0:
            movement = np.random.choice(movements)
            move = self.moves_dict[movement]
        else:
            move = [0,0]
        return move
    
    def step(self, action):
        self.current_steps += 1
        # Baseline reward
        reward = 0
        # Get grid movement
        movement = self._action_to_movement(action)
        # Compute next agent pos
        agent_pos = self._get_agent_pos()
        next_pos = agent_pos + np.asarray(movement)
        # Update state only if valid movement
        if not (self.check_boundaries(next_pos)):
            pass
        else:
            self._update_agent_pos(next_pos)
            agent_pos = next_pos
            
        beacon_center = self._get_beacon_center()
        if (beacon_center == agent_pos).all():
            reward = 1
            self._spawn_beacon()
        # Check if number of steps has exceeded the maximum for an episode
        if self.current_steps == self.max_steps:
            terminal = True
        else:
            terminal = False
             
        return copy.deepcopy(self.state), reward, terminal

    def check_boundaries(self, state):
        x_ok = (state[0] >= 0) and (state[0] < self.boundary[0])
        y_ok = (state[1] >= 0) and (state[1] < self.boundary[1])
        
        if x_ok and y_ok:
            return True
        else:
            return False
        
    def dist_to_goal(self, state):
        dx = np.abs(state[0] - self.goal[0])
        dy = np.abs(state[1] - self.goal[1])
        return dx + dy
    
    def get_optimal_action(self, show_all=False):
        if debug:
            print("self.state: ", self.state)
        optimal = np.zeros(self.n_actions)
        d0 = self.dist_to_goal(self.state)
        # consider all actions
        for action in range(self.n_actions):
            # compute for each the resulting state)
            movement = self.action_map[action]
            next_state = self.state + np.asarray(movement)
            if(self.check_boundaries(next_state)):
                # if the state is admitted -> compute the distance to the goal 
                d = self.dist_to_goal(next_state)
                # if the new distance is smaller than the old one, is an optimal action (optimal = 1.)
                if d < d0:
                    optimal[action] = 1.
                else:
                    optimal[action] = 0.
            else:
                # oterwise is not (optimal = 0)
                optimal[action] = 0.
        # once we have the vector of optimal, divide them by the sum
        probs = optimal/optimal.sum()
        if debug: 
            print("optimal: ", optimal)
            print("probs: ", probs)
        # finally sample the action and return it together with the log of the probability
        opt_action = np.random.choice(self.n_actions, p=probs)
        if show_all:
            return opt_action, probs
        else:
            return opt_action
    
    def __str__(self):
        print("Boundary: ", self.boundary)
        print("Initial position: ", self.initial)
        print("Current position: ", self.state)
        print("Goal position: ", self.goal)
        return ''
