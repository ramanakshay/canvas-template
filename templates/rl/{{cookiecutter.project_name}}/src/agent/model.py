import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from agent.network import MLP

class DiscreteActorCritic:
    def __init__(self, config):
        self.config = config.agent
        self.obs_dim, self.act_dim = self.config.obs_dim, self.config.act_dim
        self.hidden_dim = self.config.hidden_dim
        self.device = (
            self.config.device if self.config.device != "auto"
            else "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.actor = MLP(self.obs_dim, self.hidden_dim, self.act_dim).to(self.device)
        self.critic = MLP(self.obs_dim, self.hidden_dim, 1).to(self.device)

        self.grad_enabled = False

        self.weights_path = config.weights_path
        if config.from_pretrained:
            self.load_weights()

    def save_weights(self):
        torch.save(self.actor.state_dict(),  f'{self.weights_path}actor.pth')
        torch.save(self.critic.state_dict(),  f'{self.weights_path}critic.pth')

    def load_weights(self):
        self.actor.load_state_dict(torch.load(f"{self.weights_path}actor.pth", weights_only=True))
        self.critic.load_state_dict(torch.load(f"{self.weights_path}critic.pth", weights_only=True))

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
        
