import numpy as np

class OneStateGaussianMDP:
    def __init__(self):
        # initialize MDP
        self.end           = False
        self.current_state = 1
        self.num_actions   = 2
        self.num_states    = 2
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
        # left action
        else:
            # transition to B
            if self.current_state == 1:
                self.current_state += 1
                reward = 0
            # gaussian reward and terminate
            else:
                reward = np.random.normal(loc=-0.1, scale=1, size=None)
                self.end = True

        # check step counter
        if self.step_count == 1000:
            self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1
        return state, reward, self.end, {}

