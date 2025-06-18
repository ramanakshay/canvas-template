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
                _, act = self.agent.get_action(torch.from_numpy(obs))
                obs, reward, terminated, truncated, info = self.env.step(
                    act.detach().numpy()
                )
                done = terminated or truncated
                if done:
                    break

        print(f"Episode time taken: {self.env.time_queue}")
        print(f"Episode total rewards: {self.env.return_queue}")
        print(f"Episode lengths: {self.env.length_queue}")
