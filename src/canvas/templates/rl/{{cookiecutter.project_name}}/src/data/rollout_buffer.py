from data.batch import Batch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    def __init__(self, obs_space, act_space, config):
        logger.info("Initializing Rollout Buffer...")
        self.config = config
        self.capacity = self.config.capacity
        self.minibatch_size = self.config.minibatch_size
        self.num_minibatches = self.capacity // self.minibatch_size
        assert self.capacity % self.minibatch_size == 0, (
            "Buffer capacity must be divisible by the minibatch size."
        )

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
            logprob=np.empty((self.capacity, 1), dtype=np.float32),
            rew=np.empty((self.capacity, 1), dtype=np.float32),
            done=np.empty((self.capacity, 1), dtype=bool),
        )
        self.size = 0
        logger.info(f"Buffer Capacity: {self.capacity}")
        logger.info(f"Minibatch Size: {self.minibatch_size}")
        logger.info(f"Num Minibatches: {self.num_minibatches}")
        logger.info("Empty Buffer Initialized.")

    def insert(self, data):
        assert self.size < self.capacity, "Rollout Buffer is full."
        for key in data:
            self.data[key][self.size] = data[key]
        self.size += 1

    def get_batch(self):
        assert self.size == self.capacity, "Rollout Buffer is not full yet."
        return Batch(
            data=self.data, batch_size=self.capacity, minibatch_size=self.minibatch_size
        )

    def reset(self):
        self.size = 0
