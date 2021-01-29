import os
import numpy as np
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def replacing(args, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1)
    return trace

def accumulating(args, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1, reduce='add')
    return trace

def dutch(args, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1-args.alpha, reduce='multiply')
    trace = trace.scatter_(1, actions.unsqueeze(1), 1, reduce='add')
    return trace










