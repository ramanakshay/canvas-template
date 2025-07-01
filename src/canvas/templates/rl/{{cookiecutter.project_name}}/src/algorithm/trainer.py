import torch
from tqdm import tqdm
from algorithm.evaluator import Evaluator


class OnPolicyRLTrainer:
    def __init__(self, env, buffer, agent, config):
        self.config = config
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.evaluator = Evaluator(env, agent, config.evaluator)

    def run_training(self):
        env = self.env.env

        obs, info = env.reset()
        for step in range(self.buffer.capacity):
            act, logprob = self.agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(act)
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
                obs, info = env.reset()
            else:
                obs = next_obs

        batch = self.buffer.get_batch()
        self.agent.train(batch)
        self.buffer.reset()

    def run(self):
        print(f"Total Timesteps = {self.config.epochs * self.buffer.capacity}")
        for epoch in tqdm(range(self.config.epochs)):
            self.run_training()
            if epoch % self.config.eval_interval == 0:
                self.evaluator.run()
