import os
import sys
import numpy as np
import argparse
import torch
import tensorflow as tf
import pickle as pkl

from Pytorch import *
from Tensorflow import *
from MDPs import *
from utils.utils import to_np, to_tensor, to_torch
from traces.traces import replacing_trace, accumulating_trace, dutch_trace


def build parser():
    parser = argparse.ArgumentParser(description='Deep Eligibility Traces Args')
    parser.add_argument('--log_dir', type=str, default="/log/",
                        help='Directory for storing logs (default: /log/)')
    parser.add_argument('--env', type=str, default="CyclicMDP",
                        help='Toy environment (default: CyclicMDP)')
    parser.add_argument('--lib', type=str, default="torch",
                        help='Deep Learning Library to use (default: torch)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--lamb', type=float, default=0.95, metavar='G',
                        help='lambda value for trace updates (default: 0.95)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='G',
                        help='learning rate of SAC (default: 0.005)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--num_steps', type=float, default=100001, metavar='N',
                        help='maximum number of steps (default: 100000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10000, metavar='N',
                        help='save model and results every xth step (default: 10000)')

    return parser


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
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)   

    log_dict = {}
    log_dict['rewards'] = []
    log_dict['loss'] = []

    env_list = ['CyclicMDP', 'OneStateGaussianMDP', 'OneStateMDP']

    if args.env in env_list:
        env = getattr(sys.modules[__name__], args.env)
    else:
        env = gym.make(args.env)
        env.seed(args.seed)
    


if __name__=="__main__":
    main()



