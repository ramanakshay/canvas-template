import numpy as np


class RolloutBuffer:
    def __init__(self, obs_space, act_space, config):
        self.config = config
        self.capacity = self.config.capacity
        self.data = dict(
            obs=np.empty(
                (self.capacity, *obs_space.shape),
                dtype=obs_space.dtype,
            ),
            next_obs=np.empty(
                (self.capacity, *obs_space.shape),
                dtype=obs_space.dtype,
            ),
            act=np.empty((self.capacity, *act_space.shape), dtype=act_space.dtype),
            logprob=np.empty((self.capacity,), dtype=np.float32),
            rew=np.empty((self.capacity,), dtype=np.float32),
            done=np.empty((self.capacity,), dtype=bool),
        )
        self.size = 0

    def insert(self, data):
        assert self.size < self.capacity
        for key in data:
            self.data[key][self.size] = data[key]
        self.size += 1

    def reset(self):
        self.size = 0
