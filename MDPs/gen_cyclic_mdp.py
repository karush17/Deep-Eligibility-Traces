import numpy as np

class GeneralizedCyclicMDP:
    def __init__(self, num_states=3):
        # initialize MDP
        self.end           = False
        self.current_state = 1
        self.num_actions   = 3
        self.num_states    = num_states
        self.step_count = 0

    def reset(self):
        # reset MDP
        self.step_count = 0
        self.end = False
        self.current_state = 1
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.
        return state

    def step(self, action):
        # increment step count
        self.step_count += 1

        # positive reward for clockwise action
        if action == 1:
            if self.current_state !=self.num_states:
                self.current_state += 1
            else:
                self.current_state =  1
            reward = 1
        # negative reward for anticlockwise action
        else:
            reward = -1
            self.end = True

        # check step counter
        if self.step_count >= 1000:
            self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1
        return state, reward, self.end, {}

