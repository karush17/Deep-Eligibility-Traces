import os
import numpy as np
import argparse
import torch
import tensorflow as tf
import pickle as pkl

from utils.utils import *
from traces.traces import *
from Pytorch import *
from Tensorflow import *



def build parser():


def main():
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    if args.lib=='torch':
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        torch.manual_seed(args.seed)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.config.run_functions_eagerly(True)
        tf.random.set_seed(args.seed)
    
    











