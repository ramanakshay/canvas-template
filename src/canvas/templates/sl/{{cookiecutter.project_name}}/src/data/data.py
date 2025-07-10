from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


class FashionMNISTData:
    def __init__(self, config):
        logger.info("Loading data...")
        self.config = config
        # Datasets
        self.train_dataset = datasets.FashionMNIST(
            root=self.config.dataset_path,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        logger.info(f"Train Dataset Size: {len(self.train_dataset)}")

        self.val_dataset = datasets.FashionMNIST(
            root=self.config.dataset_path,
            train=False,
            download=True,
            transform=ToTensor(),
        )
        logger.info(f"Validation Dataset Size: {len(self.val_dataset)}")

        # Dataloaders
        batch_size = self.config.batch_size
        self.train_dataloader = DataLoader(self.train_dataset, batch_size)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size)
        logger.info(f"Batch Size: {self.config.batch_size}")
        logger.info("Data loaded.")
