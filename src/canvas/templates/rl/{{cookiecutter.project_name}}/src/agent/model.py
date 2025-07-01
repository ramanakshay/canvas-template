import torch
import torch.nn.functional as F
from torch import optim
from agent.network import CategoricalActor, Critic


class PPOAgent:
    def __init__(self, obs_space, act_space, config, device):
        self.config = config
        self.obs_dim, self.act_dim = obs_space.shape[0], act_space.n
        self.hidden_dims = self.config.hidden_dims

        self.actor = CategoricalActor(self.obs_dim, self.act_dim, self.hidden_dims).to(
            device
        )
        self.critic = Critic(self.obs_dim, self.hidden_dims).to(device)
        self.device = device

        self.epsilon = self.config.epsilon
        self.gamma = self.config.gamma
        self.lamda = self.config.lamda
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            self.config.learning_rate,
        )

    def act(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        dist = self.actor(obs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        action, logprob = action.detach().cpu().numpy(), logprob.detach().cpu().numpy()
        return action, logprob

    def _calculate_advantage(self, obs, next_obs, rew, done):
        size = len(obs)
        with torch.no_grad():
            values = self.critic(obs)
            next_values = self.critic(next_obs)

        # GAE estimate
        advantages = torch.empty((size,), dtype=torch.float32).to(self.device)
        advantage = 0
        for t in reversed(range(size)):
            if done[t]:
                advantage = 0
            delta = (
                rew[t] + self.gamma * (1 - done[t].int()) * next_values[t] - values[t]
            )
            advantage = delta + self.gamma * self.lamda * advantage
            advantages[t] = advantage

        normalized_advs = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return normalized_advs

    def _calculate_losses(self, obs, act, logprob, next_obs, rew, done):
        advantages = self._calculate_advantage(obs, next_obs, rew, done)
        dists = self.actor(obs)
        logprobs, old_logprobs = dists.log_prob(act), logprob

        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(obs).squeeze()
        critic_loss = (-advantages * values).mean()

        return actor_loss, critic_loss

    def train(self, batch):
        for key in batch:
            batch[key] = torch.from_numpy(batch[key]).to(self.device)

        actor_loss, critic_loss = self._calculate_losses(**batch)
        actor_loss.backward()
        critic_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = {"actor": actor_loss.item(), "critic": critic_loss.item()}

        return loss
