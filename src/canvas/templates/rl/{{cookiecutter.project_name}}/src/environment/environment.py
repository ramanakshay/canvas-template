import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class GymEnvironment:
    def __init__(self, config):
        logger.info("Building Gym Environment...")
        self.config = config
        self.env = gym.make(
            self.config.name, max_episode_steps=self.config.max_ep_steps
        )
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space
        self.max_ep_steps = self.config.max_ep_steps
        logger.info(f"Environment Name: {self.config.name}")
        logger.info(f"Max Episode Steps: {self.max_ep_steps}")
        logger.info("Gym Environment Built.")
