from data import TranslateData
from model import TranslatorModel
from algorithm import SSLTrainer

import torch
import hydra
from omegaconf import DictConfig
from rope.contrib.autoimport.models import Model


def setup(config):
    torch.manual_seed(42)
    device = config.system.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    ## SETUP ##
    device = setup(config)

    ## DATA ##
    data = TranslateData(config.data, device)
    src_vocab, tgt_vocab = len(data.vocab["de"]), len(data.vocab["en"])
    print("Data Loaded.")

    ## MODEL ##
    model = TranslatorModel(src_vocab, tgt_vocab, config.model, device)
    print("Model Created.")

    ## ALGORITHM ##
    algorithm = SSLTrainer(data, model, config.algorithm, device)
    algorithm.run()
    print("Done!")


if __name__ == "__main__":
    main()
