import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from networks.pytorch_networks import *
from utils.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SARSA(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(SARSA, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions).to(DEVICE)
        self.value_net = ValueNetwork(args, state_dims, num_actions).to(DEVICE)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.opt_value = torch.optim.Adam(self.value_net.parameters(), lr=args.lr)
        self.trace = {}
        print(self.actor)
        print(self.opt_value)
    
    def forward(self, states):
        return self.actor(states)
    
    def get_actions(self, epsilon, states):
        return self.actor.get_actions(epsilon, states)
    
    def update(self, buffer, args, states, reward, next_states, done, step_count):
        states, actions, rewards, next_states, dones = buffer.sample(args.batch_size)
        states = to_torch(states)
        actions = to_torch(actions)
        rewards = to_torch(rewards)
        next_states = to_torch(next_states)
        dones = to_torch(dones)
        vals = self.actor(states)#[self.actor.get_actions(states)]
        next_vals = self.actor(next_states)
        # print(vals)
        # print(next_vals)
        # next_vals = np.random.choice(next_vals, 1)[0]
        target = reward + self.args.gamma*next_vals*(1 - dones)
        td_error = (target.detach() - vals).pow(2).mean()
        self.opt_actor.zero_grad()
        td_error.backward()
        self.opt_actor.step()
        return td_error

    def reset_trace(self, step_count):
        if step_count==1:
            for idx, p in enumerate(self.actor.parameters()):
                self.trace[idx] = torch.zeros(p.data.shape).to(DEVICE)
        return self.trace
















