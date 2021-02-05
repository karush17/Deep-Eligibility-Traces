import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from networks.pytorch_networks import *
from utils.utils import *
from traces import torch_traces

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class SARSA(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(SARSA, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions).to(DEVICE)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.trace = torch.zeros((self.args.batch_size, self.num_actions)).to(DEVICE)
        print(self.actor)
        print(self.opt_actor)
    
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
        vals = self.actor(states)#[self.actor.get_actions(states)]
        vals = vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_vals = self.actor(next_states)
        next_actions = self.actor.get_actions(steps, next_states).type(torch.int64)
        next_vals = next_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + self.args.gamma*next_vals*(1 - dones)
        td_error = (target.detach() - vals).pow(2)
        # trace update
        if self.args.trace!='none':
            if step_count==0:
                self.trace = self.reset_trace()
            self.trace  = self.update_trace(actions)
            td_error *= self.trace.gather(1, actions.unsqueeze(1)).squeeze(1)
            self.trace *= self.args.gamma*self.args.lamb
        # td update
        td_error = td_error.mean()
        self.opt_actor.zero_grad()
        td_error.backward()
        self.opt_actor.step()
        return td_error.item()

    def update_trace(self, actions):
        return getattr(torch_traces, self.args.trace)(self.args, actions, self.trace)

    def reset_trace(self):
            return torch.zeros((self.args.batch_size, self.num_actions)).to(DEVICE)
















