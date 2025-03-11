from algorithm.train import Trainer
from model.classifier import Classifier
from data.data import FashionMNISTData

import hydra
from omegaconf import DictConfig, OmegaConf

def setup():
    # (Optional) Initial setup (can return process state (for ddp), amp context, etc.)
    # random seed
    torch.manual_seed(42)

def cleanup():
    # (Optional) Cleanup code (ex. destroy_process_group)
    pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##
    setup()
    
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

    ## CLEANUP ##
    cleanup()

if __name__ == "__main__":
    main()
