import torch
from gymnasium.wrappers import RecordEpisodeStatistics


class Evaluator:
    def __init__(self, env, agent, config):
        self.agent = agent
        self.config = config
        self.env = RecordEpisodeStatistics(
            env.env, buffer_length=self.config.eval_episodes
        )

    def run(self):
        for _ in range(self.config.eval_episodes):
            obs, info = self.env.reset()
            while True:
                act, _ = self.agent.act(obs)
                obs, reward, terminated, truncated, info = self.env.step(act)
                done = terminated or truncated
                if done:
                    break

        avg_return = sum(list(self.env.return_queue)) / len(self.env.return_queue)
        avg_length = sum(list(self.env.length_queue)) / len(self.env.length_queue)
        print(f"Evaluation over {self.config.eval_episodes} episodes:")
        print(f"Average Return: {avg_return:.2f}")
        print(f"Average Length: {avg_length:.2f}")
