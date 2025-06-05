import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm


class BasePolicyGradient:
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config.algorithm

        self.optimizer = Adam([
            {'params': self.agent.actor.parameters()},
            {'params': self.agent.critic.parameters()}
        ], self.config.learning_rate)

    def generate_episode(self):
        ep_obs, ep_acts, ep_rews = [], [], []

        obs,_ = self.env.reset()
        while True:
            ep_obs.append(obs)
            _, act = self.agent.get_action(torch.tensor(obs, dtype=torch.float))
            obs, reward, terminated, truncated, info = self.env.step(act.detach().numpy())

            ep_acts.append(act)
            ep_rews.append(reward)

            if terminated or truncated:
                break

        ep_obs = torch.tensor(np.array(ep_obs))
        ep_acts = torch.tensor(np.array(ep_acts))
        ep_rews = torch.tensor(np.array(ep_rews))
        ep_len = len(ep_obs)

        return ep_obs, ep_acts, ep_rews, ep_len

    def generate_batch(self):
        self.agent.enable_grad(False)
        timesteps_per_batch = self.config.timesteps_per_batch
        batch_obs, batch_acts, batch_rews, batch_lens = [], [], [], []
        timestep = 0
        while (timestep < timesteps_per_batch):
            ep_obs, ep_acts, ep_rews, ep_len = self.generate_episode()
            batch_obs.append(ep_obs)
            batch_acts.append(ep_acts)
            batch_rews.append(ep_rews)
            batch_lens.append(ep_len)
            timestep += ep_len

        return batch_obs, batch_acts, batch_rews, batch_lens

    def run(self):
        total_timesteps = self.config.total_timesteps
        iteration, timestep = 0, 0
        # single progress bar since no eval part
        pbar = tqdm(total=total_timesteps, unit='ts')
        while (timestep < total_timesteps):
            batch_obs, batch_acts, batch_rews, batch_lens = self.generate_batch()

            batch_sum = [ep_rewards.sum().item() for ep_rewards in batch_rews]
            avg_ep_rew = np.mean(batch_sum)
            avg_ep_len = np.mean(batch_lens)

            iteration += 1
            timestep += np.sum(batch_lens)

            self.optimizer.zero_grad()
            actor_loss, critic_loss = self.calculate_losses(batch_obs, batch_acts, batch_rews)
            actor_loss.backward()
            if critic_loss is not None: critic_loss.backward()
            self.optimizer.step()
            pbar.write((f"Iteration {iteration}: Average Return = {avg_ep_rew:.2f}, Average Episode Length = {avg_ep_len:.2f}"))

            pbar.set_description(f"Iteration {iteration}")
            pbar.set_postfix(avg_ep_rew=avg_ep_rew, avg_ep_len=avg_ep_len)
            pbar.update(timestep - pbar.n)
        pbar.close()
        self.agent.save_weights()

    def calculate_losses(self):
        raise NotImplementedError('`calculate_losses` function not implemented.')


class VanillaPolicyGradient(BasePolicyGradient):
    def __init__(self, agent, env, config):
        BasePolicyGradient.__init__(self, agent, env, config)

    def calculate_advantage(self, batch_obs, batch_rews):
        gamma = self.config.gamma
        lam = self.config.lam

        batch_lam = []
        for ep_num in range(len(batch_rews)):
            ep_obs = batch_obs[ep_num]
            ep_rew = batch_rews[ep_num]
            ep_values = self.agent.get_value(ep_obs).squeeze()
            ep_values_prime = torch.cat((ep_values[1:], torch.tensor([0.0])))
            ep_res = ep_rew + gamma * ep_values_prime - ep_values

            ep_lam = []
            residual_sum = 0
            for res in reversed(ep_res):
                residual_sum = res + lam * gamma * residual_sum
                ep_lam.insert(0, residual_sum)
            ep_lam = torch.tensor(np.array(ep_lam))
            batch_lam.append(ep_lam)

        advantages = torch.cat(batch_lam)
        normalized_advs = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return normalized_advs

    def calculate_losses(self, batch_obs, batch_acts, batch_rews):
        self.agent.enable_grad(False)
        advantages = self.calculate_advantage(batch_obs, batch_rews)

        self.agent.enable_grad(True)
        batch_obs, batch_acts = torch.cat(batch_obs), torch.cat(batch_acts)

        dists, _ = self.agent.get_action(batch_obs)
        log_probs = dists.log_prob(batch_acts)
        actor_loss = (-advantages * log_probs).mean()

        values = self.agent.get_value(batch_obs).squeeze()
        critic_loss = (-advantages * values).mean()

        return actor_loss, critic_loss






