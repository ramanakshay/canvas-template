import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from agent.network import MLP


class DiscreteActorCritic:
    def __init__(self, obs_dim, act_dim, config, device):
        self.config = config
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.hidden_dim = self.config.hidden_dim
        self.actor = MLP(self.obs_dim, self.hidden_dim, self.act_dim).to(device)
        self.critic = MLP(self.obs_dim, self.hidden_dim, 1).to(device)
        self.grad_enabled = False

    def enable_grad(self, mode):
        self.grad_enabled = mode

    def get_action(self, obs):
        with torch.set_grad_enabled(self.grad_enabled):
            outputs = self.actor(obs)
            probs = F.softmax(outputs, dim=-1)
            dist = Categorical(probs)
            act = dist.sample()
            return dist, act

    def get_value(self, obs):
        with torch.set_grad_enabled(self.grad_enabled):
            value = self.critic(obs)
            return value
