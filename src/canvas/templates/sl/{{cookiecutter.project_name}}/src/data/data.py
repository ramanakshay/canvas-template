from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class FashionMNISTData:
    def __init__(self, config):
        self.config = config

        # Datasets
        self.train_dataset = datasets.FashionMNIST(
            root=self.config.dataset_path,
            train=True,
            download=True,
            transform=ToTensor(),
        )

        self.val_dataset = datasets.FashionMNIST(
            root=self.config.dataset_path,
            train=False,
            download=True,
            transform=ToTensor(),
        )

        # Dataloaders
        batch_size = self.config.batch_size
        self.train_dataloader = DataLoader(self.train_dataset, batch_size)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size)
