import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from networks.pytorch_networks import *
from utils.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DoubleQLearning(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(DoubleQLearning, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions).to(DEVICE)
        self.target_actor = ActorNetwork(args, state_dims, num_actions).to(DEVICE)
        self.value_net = ValueNetwork(args, state_dims, num_actions).to(DEVICE)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.opt_value = torch.optim.Adam(self.value_net.parameters(), lr=args.lr)
        self.trace = {}
        print(self.actor)
        print(self.opt_value)
    
    def forward(self, states):
        return self.actor(states)
    
    def get_actions(self, steps, states):
        return self.actor.get_actions(steps, states)
    
    def update(self, replay_buffer, steps, step_count):
        batch_size = self.args.batch_size
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = to_torch(states)
        actions = to_torch(actions).type(torch.int64)
        rewards = to_torch(rewards)
        next_states = to_torch(next_states)
        dones = to_torch(dones)
        vals = self.actor(states)
        vals = vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_vals = self.actor(next_states)
        next_q_vals = self.target_actor(next_states)
        next_q_values = next_q_vals.gather(1, torch.max(next_vals, 1)[1].unsqueeze(1)).squeeze(1)
        target = reward + self.args.gamma*next_q_values*(1 - dones)
        td_error = (target.detach() - vals).pow(2).mean()
        self.opt_actor.zero_grad()
        td_error.backward()
        self.opt_actor.step()
        self.update_target(step_count)
        return td_error

    def update_target(self, step_count):
        if step_count==1:
            self.target_actor.load_state_dict(self.actor.state_dict())

    def reset_trace(self, step_count):
        if step_count==1:
            for idx, p in enumerate(self.actor.parameters()):
                self.trace[idx] = torch.zeros(p.data.shape).to(DEVICE)
        return self.trace
















