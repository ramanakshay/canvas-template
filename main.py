from algorithm.algorithm import Algorithm
from agent.model import Model

from torchvision import datasets
from torchvision.transforms import ToTensor

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATASET ##
    dataset = {
        "train": datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()),
        "test": datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor())
    }

    ## MODEL ##
    model = Model(config.agent)

    ## ALGORITHM ##
    alg = Algorithm(dataset, model, config.algorithm)
    alg.run()

if __name__ == "__main__":
    main()
