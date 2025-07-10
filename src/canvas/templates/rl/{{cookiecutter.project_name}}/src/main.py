from environment import GymEnvironment
from data import RolloutBuffer
from agent import PPOAgent
from algorithm import RLTrainer

import torch
import hydra
from omegaconf import DictConfig


def setup(config):
    torch.manual_seed(42)
    device = torch.device(config.device)
    return device


def cleanup():
    pass


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    # --- SETUP ---
    device = setup(config.system)

    # --- ENVIRONMENT ---
    env = GymEnvironment(config.environment)

    # --- DATA ---
    buffer = RolloutBuffer(env.obs_space, env.act_space, config.buffer)

    # --- AGENT ---
    agent = PPOAgent(env.obs_space, env.act_space, config.agent, device)

    # --- ALGORITHM ---
    trainer = RLTrainer(env, buffer, agent, config.trainer)
    trainer.run()

    # --- CLEANUP ---
    cleanup()


if __name__ == "__main__":
    main()
