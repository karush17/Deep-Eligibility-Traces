import os
import numpy as np
import torch

def replacing(args, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1)
    return trace

def accumulating(args, actions, trace):
    upd = torch.zeros(trace.shape).to(DEVICE)
    trace += upd.scatter_(1, actions.unsqueeze(1), 1)
    return trace

def dutch(args, state, trace):
    trace = args.gamma*args.lamb*trace + (1 - args.alpha*args.gamma*args.lamb*trace.T*state)*state
    return trace










