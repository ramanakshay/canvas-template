from agent.model import DiscreteActorCritic
from algorithm.train import VanillaPolicyGradient
import gymnasium as gym

import hydra
from omegaconf import DictConfig, OmegaConf


def setup(config):
    torch.manual_seed(42)
    device = config.system.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##
    device = setup(config)

    ## ENVIRONMENT ##
    env = gym.make(config.env.name, max_episode_steps = config.env.max_ep_steps)
    print("Environment Built.")

    ## AGENT ##
    agent = DiscreteActorCritic(config)
    print('Agent Created.')

    ## ALGORITHM ##
    print('Algorithm Running.')
    alg = VanillaPolicyGradient(agent, env, config)
    alg.run()
    print('Done!')

if __name__ == "__main__":
    main()