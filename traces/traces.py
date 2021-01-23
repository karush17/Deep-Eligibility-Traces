import os
import numpy as np


def replacing_trace(args, state, trace):
    trace[state] = 1
    trace[~state] *= args.gamma*args.lamb
    return trace

def accumulating_trace(args, state, trace):
    trace += args.gamma*args.lamb
    trace[state] += 1
    return trace

def dutch_trace(args, state, trace):
    trace = args.gamma*args.lamb*trace + (1 - args.alpha*args.gamma*args.lamb*trace.T*state)*state
    return trace










