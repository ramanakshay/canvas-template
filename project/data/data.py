from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def load_data(config):
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

    batch_size = config.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size)

    dataloaders = {
        'train': train_dataloader,
        'test': test_dataloader
    }

    return dataloaders