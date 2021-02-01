import numpy as np
import random

class MultiChainMDP:
    def __init__(self, num_int_states=16):
        # initialize MDP
        self.end           = False
        self.current_state = 1
        self.num_chains = 5 # number of parallel chains
        self.current_chain = np.zeros(self.num_chains) # current chain
        self.num_actions   = 2
        self.p = 0.25 # prob of +1 reward
        self.num_int_states = num_int_states # number of states in each chain
        self.int_state = np.zeros(self.num_int_states)
        self.num_states    = num_int_states + self.num_chains # total number of states
        self.step_count = 0

    def reset(self):
        # reset MDP
        self.step_count = 0
        self.end = False
        self.current_state = 1
        self.current_chain = np.zeros(self.num_chains) # position in the chain
        self.int_state = np.zeros(self.num_int_states) # selected chain
        state = np.concatenate((self.int_state, self.current_chain), axis=0)
        return state

    def step(self, action):
        self.step_count += 1 
        if action==1:
            # first white node
            if self.current_state==1:
                self.current_chain[np.random.randint(low=0, high=self.num_chains)] = 1 # randomly select chain
                self.int_state[self.current_state-1] = 1 # move to first node of chain
                state = np.concatenate((self.int_state, self.current_chain), axis=0) # complete state vector
                reward = 1
                self.current_state += 1
            # last state of current chain
            elif self.current_state==self.num_int_states+1:
                self.current_chain = np.zeros(self.num_chains) # exit current chain
                self.int_state = np.zeros(self.num_int_states) # move to orange node
                state = np.concatenate((self.int_state, self.current_chain), axis=0)
                reward = -1 if random.random() < self.p else 1 # -1 with prob p, 1 with prob (1-p)
                self.end = True
            # move along the chain
            else:
                self.int_state = np.zeros(self.num_int_states)
                self.int_state[self.current_state-1] = 1 # next node of the chain
                state = np.concatenate((self.int_state, self.current_chain), axis=0)
                reward = 1
                self.current_state += 1

        # negative reward for backward action
        else:
            state = np.concatenate((self.int_state, self.current_chain), axis=0)
            reward = -1
            self.end = True

        # check step counter
        if self.step_count >= 1000:
            self.end = True

        print(state.shape)
        print(self.int_state.shape)
        print(self.current_chain.shape)
        print(self.current_state)
        return state, reward, self.end, {}

