from algorithm.train import Trainer
from model.classifier import Classifier

from torchvision import datasets
from torchvision.transforms import ToTensor

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATASET ##
    dataset = {
        "train": datasets.MNIST(
            root=config.data.path,
            train=True,
            download=True,
            transform=ToTensor()),
        "test": datasets.MNIST(
            root=config.data.path,
            train=False,
            download=True,
            transform=ToTensor())
    }

    ## MODEL ##
    model = Classifier(config.model)

    ## ALGORITHM ##
    alg = Trainer(dataset, model, config.algorithm)
    alg.run()

if __name__ == "__main__":
    main()
