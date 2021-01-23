import numpy as np
import torch
import torch.nn as nn
from utils.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ActorNetwork(nn.Module):
    def __init__(self, args, num_inputs, num_actions):
        super(ActorNetwork, self).__init__()
        self.l1 = nn.Linear(num_inputs, 128)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(128, num_actions)

    def forward(self, states):
        x = to_torch(states.to(DEVICE))
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        x = self.l4(x)
        return x
    
    def get_actions(self, states):
        x = self.forward(states.to(DEVICE))
        return np.squeeze(x, axis=-1)


class ValueNetwork(nn.Module):
    def __init__(self, args, num_inputs, num_actions):
        super(ValueNetwork, self).__init__()
        self.l1 = nn.Linear(num_inputs, 128)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(128, 128)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(128, 1)

    def forward(self, states):
        x = to_torch(states)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        x = self.l4(x)
        return x


