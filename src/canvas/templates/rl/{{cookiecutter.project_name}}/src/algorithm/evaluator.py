import logging
from gymnasium.wrappers import RecordEpisodeStatistics

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, env, agent, config):
        self.config = config
        self.agent = agent
        self.env = env

    def run(self):
        env = RecordEpisodeStatistics(
            self.env.env, buffer_length=self.config.eval_episodes
        )

        for _ in range(self.config.eval_episodes):
            obs, info = env.reset()
            while True:
                act, _ = self.agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(act)
                done = terminated or truncated
                if done:
                    break

        avg_return = sum(list(env.return_queue)) / len(env.return_queue)
        avg_length = sum(list(env.length_queue)) / len(env.length_queue)

        logger.info(f"Evaluation over {self.config.eval_episodes} episodes.")
        logger.info(
            f"Average Return: {avg_return:.2f}, Average Length: {avg_length:.2f}"
        )
