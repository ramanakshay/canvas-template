import torch
import torch.nn.functional as F
from torch import optim
from agent.network import CategoricalActor, Critic
import logging
import os

logger = logging.getLogger(__name__)


class PPOAgent:
    def __init__(self, obs_space, act_space, config, device):
        logger.info("Creating agent...")
        self.config = config
        self.device = device
        self.obs_dim, self.act_dim = obs_space.shape[0], act_space.n
        logger.info(f"Observation Dimension: {self.obs_dim}")
        logger.info(f"Action Dimension: {self.act_dim}")

        self.actor = CategoricalActor(
            self.obs_dim, self.act_dim, self.config.actor.hidden_dims
        ).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.actor.optimizer.learning_rate
        )

        self.critic = Critic(self.obs_dim, self.config.critic.hidden_dims).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.critic.optimizer.learning_rate
        )
        logger.info("Agent created.")

    def act(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample()
            logprob = dist.log_prob(action)
        action, logprob = (
            action.squeeze(0).detach().cpu().numpy(),
            logprob.squeeze(0).detach().cpu().numpy(),
        )
        return action, logprob

    def gae_estimation(self, batch):
        obs, next_obs = batch["obs"], batch["next_obs"]
        done = batch["done"]
        rew = batch["rew"]

        batch_size = len(obs)
        chunk_size = self.config.gae_chunk_size or batch_size

        values = torch.zeros(batch_size, 1)
        next_values = torch.zeros(batch_size, 1)

        with torch.no_grad():
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                obs_chunk = obs[start:end].to(self.device)
                next_obs_chunk = next_obs[start:end].to(self.device)
                values[start:end] = self.critic(obs_chunk).cpu()
                next_values[start:end] = self.critic(next_obs_chunk).cpu()

        # GAE estimate using a reverse loop
        deltas = rew + self.config.gamma * (~done) * next_values - values
        advantages = torch.zeros_like(deltas)
        advantage = 0
        for t in reversed(range(batch_size)):
            advantage = deltas[t] + self.config.gamma * self.config.gae_lambda * advantage * ~done[t]
            advantages[t] = advantage

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        return advantages, returns

    def loss(self, batch):
        obs, act, logprob = (
            batch["obs"].to(self.device),
            batch["act"].to(self.device),
            batch["logprob"].to(self.device),
        )
        advantages, returns = (
            batch["advantages"].to(self.device),
            batch["returns"].to(self.device),
        )

        dists = self.actor(obs)
        logprobs, old_logprobs = dists.log_prob(act).unsqueeze(-1), logprob

        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        values = self.critic(obs)
        critic_loss = F.mse_loss(values, returns)

        return actor_loss, critic_loss

    def train(self, batch):
        advantages, returns = self.gae_estimation(batch.data)
        batch.data["advantages"] = advantages
        batch.data["returns"] = returns

        avg_actor_loss, avg_critic_loss = 0.0, 0.0
        for minibatch in batch:
            actor_loss, critic_loss = self.loss(minibatch)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss = actor_loss + self.config.critic_coef * critic_loss
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            avg_actor_loss += actor_loss.item()
            avg_critic_loss += critic_loss.item()

        avg_actor_loss /= batch.num_minibatches
        avg_critic_loss /= batch.num_minibatches

        losses = {"actor": avg_actor_loss, "critic": avg_critic_loss}
        return losses

    def save(self):
        save_path = self.config.checkpoint_path
        logger.info(f"Saving model to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            save_path,
        )

    def load(self):
        load_path = self.config.checkpoint_path
        if os.path.exists(load_path):
            logger.info(f"Loading model from {load_path}")
            checkpoint = torch.load(load_path, map_location=self.device)
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
        else:
            logger.error(f"Checkpoint path not found: {load_path}")
            raise FileNotFoundError(f"Checkpoint path not found: {load_path}")
