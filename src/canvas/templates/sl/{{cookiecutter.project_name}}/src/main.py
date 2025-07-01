from algorithm import SLTrainer
from model import ClassifierModel
from data import FashionMNISTData

import torch
import hydra
from omegaconf import DictConfig


def setup(config):
    torch.manual_seed(42)
    device = config.device
    return device


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    ## SETUP ##
    device = setup(config.system)

    ## DATA ##
    data = FashionMNISTData(config.data)
    print("Data Loaded.")

    ## MODEL ##
    model = ClassifierModel(config.model, device)
    print("Model Created.")

    ## ALGORITHM ##
    print("Running Algorithm.")
    alg = SLTrainer(data, model, config.trainer)
    alg.run()
    print("Done!")


if __name__ == "__main__":
    main()
