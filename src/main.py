from algorithm.train import Trainer
from model import Classifier
from data import FashionMNISTData
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

def setup(config):
    torch.manual_seed(42)
    device = config.system.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


# def cleanup():
#     pass

def cleanup():
    # (Optional) Cleanup code (ex. destroy_process_group)
    pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## SETUP ##
    device = setup(config)
    ## DATA ##
    data = FashionMNISTData(config)
    print('Data Loaded.')

    ## MODEL ##
    model = Classifier(config, device)
    print('Model Created.')

    ## ALGORITHM ##
    print('Running Algorithm.')
    alg = Trainer(data, model, config, device)
    alg.run()
    print('Done!')

    ## CLEANUP ##
    cleanup()

if __name__ == "__main__":
    main()
