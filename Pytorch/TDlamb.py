import os
import numpy as np
import torch
import torch.nn as nn
from networks.pytorch_networks import *
from utils.utils import *

class TDLambda(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(TDLambda, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions)
        self.value_net = ValueNetwork(args, state_dims, num_actions)
        self.opt = nn.Adam()
    
    def forward(self, states):
        return self.actor(states)
    
    def get_actions(self, states):
        return self.actor.get_actions(states)
    
    def update(self):

















