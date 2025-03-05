from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class FashionMNISTData:
    def __init__(self, config):
        self.config = config.data
        train_dataset = datasets.FashionMNIST(
            root=config.path,
            train=True,
            download=True,
            transform=ToTensor())

        test_dataset = datasets.FashionMNIST(
            root=config.path,
            train=False,
            download=True,
            transform=ToTensor())

        batch_size = self.config.batch_size

        self.train_dataloader = DataLoader(train_dataset, batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size)

    def get_dataloaders(self):
        return {'train': self.train_dataloader, 'test': self.test_dataloader}

