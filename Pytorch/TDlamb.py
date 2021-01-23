import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as ag
from networks.pytorch_networks import *
from utils.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TDLambda(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(TDLambda, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions)
        self.value_net = ValueNetwork(args, state_dims, num_actions)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.opt_value = torch.optim.Adam(self.value_net.parameters(), lr=args.lr)
        self.trace = torch.zeros(len(self.value_net.state_dict().items())).to(DEVICE)
        print(self.value_net)
        print(self.opt_value)
    
    def forward(self, states):
        return self.actor(states)
    
    def get_actions(self, states):
        return self.actor.get_actions(states)
    
    def update(self, args, states, reward, next_states, step_count):
        opt_value.zero_grad()
        vals = self.value_net(states)
        td_error = reward + self.args.gamma*self.value_net(next_states) - vals
        self.trace = self.reset_trace(step_count) 
        for idx, p in enumerate(self.value_net.parameters()):
            self.trace[idx] = self.args.gamma*self.args.lamb*self.trace[idx] + ag.grad(vals, p)
            p.grad = torch.FloatTensor(-td_error*self.trace[idx]).clone()
        opt_value.step()
        return td_error

    def reset_trace(self, step_count):
        if step_count==0:
            self.trace = torch.zeros(len(self.value_net.state_dict().items())).to(DEVICE)
        return self.trace

















