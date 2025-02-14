from algorithm.train import Trainer
from model.classifier import Classifier
from data.data import FashionMNISTData

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATA ##
    data = FashionMNISTData(config)
    print('Data Loaded.')

    ## MODEL ##
    model = Classifier(config)
    print('Model Created.')

    ## ALGORITHM ##
    print('Running Algorithm.')
    alg = Trainer(data, model, config)
    alg.run()
    print('Done!')

if __name__ == "__main__":
    main()
