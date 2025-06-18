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
    env = GymEnvironment(config.environment)
    obs_dim, act_dim = env.obs_space.shape[0], env.act_space.n
    print("Environment Built.")

    ## DATA ##
    data = RolloutBuffer(env.obs_space, env.act_space, config.data)
    print("Empty Buffer Initialized.")

    ## AGENT ##
    agent = DiscreteActorCritic(obs_dim, act_dim, config.agent, device)
    print("Agent Created.")

    ## ALGORITHM ##
    print("Algorithm Running.")
    evaluator = Evaluator(env, agent, config.algorithm.evaluator)
    trainer = PPOTrainer(env, data, agent, evaluator, config.algorithm.trainer, device)
    trainer.run()
    print("Done!")


if __name__ == "__main__":
    main()
