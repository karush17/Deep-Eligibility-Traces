import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from networks.pytorch_networks import *
from utils.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class WatkinsQ(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(WatkinsQ, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions).to(DEVICE)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.trace = {}
        for idx, p in enumerate(self.actor.parameters()):
            self.trace[idx] = torch.zeros(p.data.shape).to(DEVICE)
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
        next_vals = self.actor(next_states).max(1)[0]
        # new_next_vals = torch.zeros([self.args.batch_size,1], dtype=torch.float32).to(DEVICE)
        # for i in range(len(next_vals)):
        #     new_next_vals[i] = to_torch(np.random.choice(to_np(self.args, next_vals[i,:]), p=to_np(self.args, F.softmax(next_vals[i,:]))))
        target = rewards + self.args.gamma*next_vals*(1 - dones)
        td_error = (target.detach() - vals).pow(2).mean()
        self.opt_actor.zero_grad()
        self.trace = self.reset_trace(step_count) 
        eval_gradients = ag.grad(td_error, self.actor.parameters())
        for idx, p in enumerate(self.actor.parameters()):
            self.trace[idx] = self.args.gamma*self.args.lamb*self.trace[idx] + eval_gradients[idx]
            p.grad = td_error*self.trace[idx]
        self.opt_actor.step()
        return td_error.item()

    def reset_trace(self, step_count):
        if step_count==1:
            for idx, p in enumerate(self.actor.parameters()):
                self.trace[idx] = torch.zeros(p.data.shape).to(DEVICE)
        return self.trace
















