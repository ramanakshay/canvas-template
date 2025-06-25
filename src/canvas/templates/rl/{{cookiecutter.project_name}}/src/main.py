from environment import GymEnvironment
from data import RolloutBuffer
from agent import PPOAgent
from algorithm import OnPolicyRLTrainer

import torch
import hydra
from omegaconf import DictConfig


def setup(config):
    torch.manual_seed(42)
    device = torch.device(config.system.device)
    return device


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    ## SETUP ##
    device = setup(config)

    ## ENVIRONMENT ##
    env = GymEnvironment(config.environment)
    print("Environment Built.")

    ## BUFFER ##
    buffer = RolloutBuffer(env.obs_space, env.act_space, config.buffer)
    print("Empty Buffer Initialized.")

    ## AGENT ##
    agent = PPOAgent(env.obs_space, env.act_space, config.agent, device)
    print("Agent Created.")

    ## ALGORITHM ##
    print("Algorithm Running.")
    trainer = OnPolicyRLTrainer(env, buffer, agent, config.trainer)
    trainer.run()
    print("Done!")


if __name__ == "__main__":
    main()
