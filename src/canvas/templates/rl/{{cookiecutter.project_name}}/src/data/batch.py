import torch


class Batch:
    def __init__(self, data, batch_size, minibatch_size):
        self.data = {key: torch.tensor(val) for key, val in data.items()}
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by "
                f"minibatch_size ({self.minibatch_size})"
            )
        self.num_minibatches = self.batch_size // self.minibatch_size

    def __iter__(self):
        batch_inds = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            minibatch_inds = batch_inds[start:end]
            minibatch = dict()
            for key in self.data:
                minibatch[key] = self.data[key][minibatch_inds]
            yield minibatch
