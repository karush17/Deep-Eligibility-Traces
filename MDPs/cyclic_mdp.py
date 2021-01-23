import numpy as np

class CyclicMDP:
    def __init__(self):
        # initialize MDP
        self.end           = False
        self.current_state = 1
        self.num_actions   = 3
        self.num_states    = 3
        self.step_count = 0

    def reset(self):
        # reset MDP
        self.count_step = 0
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
            if self.current_state in [1,2]:
                self.current_state += 1
            else:
                self.current_state =  1
            reward = 1
        # negative reward for anticlockwise action
        else:
            reward = -1
            self.end = True

        # check step counter
        if self.step_count == 1000:
            self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1
        return state, reward, self.end, {}

