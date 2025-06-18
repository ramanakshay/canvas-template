import gymnasium as gym


class GymEnvironment:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(
            self.config.name, max_episode_steps=self.config.max_ep_steps
        )
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space
        self.max_ep_steps = self.config.max_ep_steps
