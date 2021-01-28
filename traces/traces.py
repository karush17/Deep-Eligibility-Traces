import os
import numpy as np


def replacing(args, state, trace):
    trace[state] = 1
    trace[~state] *= args.gamma*args.lamb
    return trace

def accumulating(args, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1)
    return trace

def dutch(args, state, trace):
    trace = args.gamma*args.lamb*trace + (1 - args.alpha*args.gamma*args.lamb*trace.T*state)*state
    return trace










