import torch
import numpy as np
from torch import optim
from algorithm.trainer.base import OnPolicyTrainer


class VPGTrainer(OnPolicyTrainer):
    def __init__(self, env, buffer, agent, evaluator, config, device):
        OnPolicyTrainer.__init__(self, env, buffer, agent, evaluator, config, device)

        self.gamma = self.config.gamma
        self.lamda = self.config.lamda
        self.optimizer = optim.Adam(
            [
                {"params": self.agent.actor.parameters()},
                {"params": self.agent.critic.parameters()},
            ],
            self.config.learning_rate,
        )

    def calculate_advantage(self, batch):
        size = len(batch["obs"])
        obs = torch.from_numpy(batch["obs"])
        next_obs = torch.from_numpy(batch["next_obs"])
        values = self.agent.get_value(obs)
        next_values = self.agent.get_value(next_obs)

        # GAE estimate
        rew, done = batch["rew"], batch["done"]
        advantages = np.empty((size,), dtype=np.float32)
        advantage = 0
        for t in zip(reversed(range(size))):
            if done[t]:
                advantage = 0
            delta = rew[t] + self.gamma * next_values[t] - values[t]
            advantage = delta + self.gamma * self.lamda * advantage
            advantages[t] = advantage

        advantages = torch.from_numpy(advantages)
        normalized_advs = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return normalized_advs

    def calculate_losses(self, batch):
        self.agent.enable_grad(False)
        advantages = self.calculate_advantage(batch)

        self.agent.enable_grad(True)
        obs, act = torch.from_numpy(batch["obs"]), torch.from_numpy(batch["act"])

        dists, _ = self.agent.get_action(obs)
        log_probs = dists.log_prob(act)
        actor_loss = (-advantages * log_probs).mean()

        values = self.agent.get_value(obs).squeeze()
        critic_loss = (-advantages * values).mean()

        return actor_loss, critic_loss

    def update(self, batch):
        actor_loss, critic_loss = self.calculate_losses(batch)
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
