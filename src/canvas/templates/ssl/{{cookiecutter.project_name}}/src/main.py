from algorithm import SSLTrainer
from model import TransformerModel
from data import TranslateData

import hydra
from omegaconf import DictConfig, OmegaConf
import torch


def setup(config):
    device = torch.device(config.device)
    return device


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    ## SETUP ##
    device = setup(config.system)

    ## DATA ##
    data = TranslateData(config.data)
    print("Data Loaded.")

    ## MODEL ##
    src_vocab, tgt_vocab = len(data.vocab["de"]), len(data.vocab["en"])
    model = TransformerModel(src_vocab, tgt_vocab, config.model, device)
    print("Model Created.")

    ## ALGORITHM ##
    print("Running Algorithm...")
    algorithm = SSLTrainer(data, model, config.trainer)
    algorithm.run()
    print("Done!")


if __name__ == "__main__":
    main()
