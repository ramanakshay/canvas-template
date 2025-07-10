from algorithm import SLTrainer
from model import ClassifierModel
from data import FashionMNISTData

import logging
import torch
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def setup(config):
    """Set up the environment."""
    torch.manual_seed(42)
    device = torch.device(config.device)
    logger.info(f"Device: {device}")
    return device


def cleanup():
    pass


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    """Main function to run the training pipeline."""
    # --- SETUP ---
    device = setup(config.system)

    # --- DATA ---
    data = FashionMNISTData(config.data)

    # --- MODEL ---
    model = ClassifierModel(config.model, device)

    # --- ALGORITHM ---
    trainer = SLTrainer(data, model, config.trainer)
    trainer.run()

    # --- CLEANUP ---
    cleanup()


if __name__ == "__main__":
    main()
