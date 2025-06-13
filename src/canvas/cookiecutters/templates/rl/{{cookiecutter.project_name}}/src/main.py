from environment import GymEnvironment
from data import RolloutBuffer
from agent import DiscreteActorCritic
from algorithm import PPOTrainer, Evaluator

import torch
import hydra
from omegaconf import DictConfig


def setup(config):
    torch.manual_seed(42)
    device = config.system.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    ## SETUP ##
    device = setup(config)

    ## ENVIRONMENT ##
    env = GymEnvironment(config)
    print("Environment Built.")

    ## DATA ##
    buffer = RolloutBuffer(env.obs_space, env.act_space, config)
    print("Empty Buffer Initialized.")

    ## AGENT ##
    agent = DiscreteActorCritic(env.obs_space, env.act_space, config, device)
    print("Agent Created.")

    ## ALGORITHM ##
    print("Algorithm Running.")
    evaluator = Evaluator(env, agent, config)
    trainer = PPOTrainer(env, buffer, agent, evaluator, config, device)
    trainer.run()
    print("Done!")


if __name__ == "__main__":
    main()
