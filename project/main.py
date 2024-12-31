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
        "train": datasets.FashionMNIST(
            root=config.data.path,
            train=True,
            download=True,
            transform=ToTensor()),
        "test": datasets.FashionMNIST(
            root=config.data.path,
            train=False,
            download=True,
            transform=ToTensor())
    }
    print('Dataset Loaded.')

    ## MODEL ##
    model = Classifier(config.model)
    print('Model Created.')

    ## ALGORITHM ##
    print('Running Algorithm.')
    alg = Trainer(dataset, model, config.algorithm)
    alg.run()
    print('Done!')

if __name__ == "__main__":
    main()
