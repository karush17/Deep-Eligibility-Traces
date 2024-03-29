import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from networks.pytorch_networks import *
from utils.utils import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ExpectedTrace(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(ExpectedTrace, self).__init__()
        self.args = args
        self.eta = 0.95
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions).to(DEVICE)
        self.exp_trace_param = torch.rand((self.num_actions, self.state_dims), requires_grad=True).to(DEVICE)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        # self.opt_trace = torch.optim.Adam(self.exp_trace_param, lr=args.lr)
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
        vals = self.actor(states)
        vals = vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_vals = self.actor(next_states).max(1)[0]
        # expected trace update- The paper does not backpropagate trace gradients into feature representations. We follow this convention but reduce one dimension in the linear approximation since we take the expectation over batch.
        ind_trace = list(self.trace.keys())[-1]
        self.exp_trace = torch.matmul(self.exp_trace_param, states.transpose(0,1))
        trace_loss = (self.exp_trace.mean(1) - self.trace[ind_trace]).pow(2).mean() # learn expectation from last layer
        trace_grad = ag.grad(trace_loss, self.exp_trace_param)[0]
        self.exp_trace_param = self.exp_trace_param + self.args.lr*trace_grad
        self.trace[ind_trace] = (1-self.eta)*self.exp_trace.mean(1) + self.eta*self.trace[ind_trace]
        # TD update
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
















