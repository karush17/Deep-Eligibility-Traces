import numpy as np
import random
import torch
import torch.nn as nn
from utils.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ActorNetwork(nn.Module):
    def __init__(self, args, num_inputs, num_actions):
        super(ActorNetwork, self).__init__()
        self.num_actions = num_actions
        self.args = args
        self.l1 = nn.Linear(num_inputs, 128)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(128, num_actions)

    def forward(self, states):
        x = to_torch(states)
        x = to_torch(x)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x
    
    def get_actions(self, steps, states):
        if random.random() > epsilon_by_step(self.args, steps):
            x = to_torch(states)
            x = self.forward(x)
            x = to_np(self.args, x)
            return np.squeeze(np.argmax(x, axis=0), axis=-1)
        else:
            return random.randrange(self.num_actions)


class ValueNetwork(nn.Module):
    def __init__(self, args, num_inputs, num_actions):
        super(ValueNetwork, self).__init__()
        self.args = args
        self.l1 = nn.Linear(num_inputs, 128)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(128, 1)

    def forward(self, states):
        x = to_torch(states)
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        return x


