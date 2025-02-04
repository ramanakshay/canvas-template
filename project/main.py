from algorithm.train import Trainer
from model.classifier import Classifier
from data.data import load_data

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## DATA ##
    dataloaders = load_data(config.data)
    print('Data Loaded.')

    ## MODEL ##
    model = Classifier(config.model)
    print('Model Created.')

    ## ALGORITHM ##
    print('Running Algorithm.')
    alg = Trainer(dataloaders, model, config.algorithm)
    alg.run()
    print('Done!')

if __name__ == "__main__":
    main()
