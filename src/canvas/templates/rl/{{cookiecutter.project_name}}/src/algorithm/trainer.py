import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from algorithm.evaluator import Evaluator

logger = logging.getLogger(__name__)


class RLTrainer:
    def __init__(self, env, buffer, agent, config):
        self.config = config
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.evaluator = Evaluator(env, agent, config.evaluator)

    def run_training(self):
        env = self.env.env
        obs, info = env.reset()
        actor_loss, critic_loss = 0.0, 0.0
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
                    done=terminated,
                )
            )
            if done:
                obs, info = env.reset()
            else:
                obs = next_obs

        num_updates = self.config.num_updates
        for _ in range(num_updates):
            batch = self.buffer.get_batch()
            loss = self.agent.train(batch)
            actor_loss += loss["actor"]
            critic_loss += loss["critic"]
        self.buffer.reset()

        actor_loss /= num_updates
        critic_loss /= num_updates

        metrics = {"actor_loss": actor_loss, "critic_loss": critic_loss}
        return metrics

    def run_evaluation(self):
        self.evaluator.run()

    def run(self):
        logger.info("Starting training...")
        total_steps = self.config.epochs * self.buffer.capacity
        logger.info(f"Num Epochs: {self.config.epochs}")
        logger.info(f"Num Updates per Epoch: {self.config.num_updates}")
        logger.info(f"Total Steps: {total_steps}")

        with (
            logging_redirect_tqdm(),
            tqdm(
                total=self.config.epochs, desc="Training Progress", unit="epoch"
            ) as progress_bar,
        ):
            for epoch in range(1, self.config.epochs + 1):
                metrics = self.run_training()
                if epoch % self.config.log_interval == 0:
                    actor_loss, critic_loss = (
                        metrics["actor_loss"],
                        metrics["critic_loss"],
                    )
                    logger.debug(
                        f"Epoch {epoch} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}"
                    )
                if epoch % self.config.eval_interval == 0:
                    logger.info(f"Epoch {epoch}")
                    self.run_evaluation()
                progress_bar.update(1)

        self.agent.save()
        logger.info("Training completed.")
