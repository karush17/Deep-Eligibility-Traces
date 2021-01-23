import numpy as np

class OneStateMDP:
    def __init__(self):
        # initialize MDP
        self.end           = False
        self.current_state = 1
        self.num_actions   = 2
        self.num_states    = 1
        self.p_right       = 0.9
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

        # right action leads to termination
        if action==1:
            reward = 0
            self.end = True
        # left action yields reward with prob 0.1
        else:
            if np.random.rand() >= self.p_right:
                reward = 1
                self.end = True
            else:
                reward = 0

        # check step counter
        if self.step_count == 1000:
            self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1
        return state, reward, self.end, {}

