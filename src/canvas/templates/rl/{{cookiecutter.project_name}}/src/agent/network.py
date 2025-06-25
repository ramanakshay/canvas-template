import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims):
        super().__init__()
        self.network = MLP(obs_dim, act_dim, hidden_dims)

    def forward(self, obs):
        x = self.network(obs)
        probs = F.softmax(x, dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims):
        super().__init__()
        self.network = MLP(obs_dim, 1, hidden_dims)

    def forward(self, obs):
        x = self.network(obs)
        return x
