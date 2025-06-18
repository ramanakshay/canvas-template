import torch
from tqdm import tqdm


class OnPolicyTrainer:
    def __init__(self, env, buffer, agent, evaluator, config, device):
        self.config = config
        self.env = env.env
        self.agent = agent
        self.buffer = buffer
        self.evaluator = evaluator
        self.device = device

    def run_epoch(self):
        obs, info = self.env.reset()
        for step in range(self.buffer.capacity):
            dist, act = self.agent.get_action(torch.from_numpy(obs))
            logprob = dist.log_prob(act).detach()
            next_obs, reward, terminated, truncated, info = self.env.step(
                act.detach().numpy()
            )
            done = terminated or truncated
            self.buffer.insert(
                dict(
                    obs=obs,
                    next_obs=next_obs,
                    act=act,
                    logprob=logprob,
                    rew=reward,
                    done=done,
                )
            )
            if done:
                obs, info = self.env.reset()
            else:
                obs = next_obs
        self.update(self.buffer.data)
        self.buffer.reset()

    def run(self):
        print(f"Total Timesteps = {self.config.epochs * self.buffer.capacity}")
        for epoch in tqdm(range(self.config.epochs)):
            self.run_epoch()
            if epoch % self.config.eval_interval == 0:
                self.evaluator.run()

    def update(self, batch):
        raise NotImplementedError("`update` function not implemented.")
