import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm


class Trainer(object):
    def __init__(self, dataset, model, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        self.batch_size = self.config.batch_size
        self.train_dataloader = DataLoader(self.dataset["train"], batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.dataset["test"], batch_size=self.batch_size)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.network.parameters(),
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)

    def train(self):
        size = len(self.dataset["train"])
        self.model.enable_grad(True)
        pbar = tqdm(self.train_dataloader)
        pbar.set_description('Train')
        for batch, (X, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            y = y.to(self.model.device)
            pred = self.model.predict(X)
            loss = self.loss_function(pred, y)
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix(loss=loss.item())

        pbar.close()


    def test(self):
        size = len(self.dataset['test'])
        self.model.enable_grad(False)
        pbar = tqdm(self.test_dataloader)
        pbar.set_description('Test')
        test_loss, test_correct = 0.0, 0.0
        for X, y in pbar:
            y = y.to(self.model.device)
            pred = self.model.predict(X)
            loss = self.loss_function(pred, y)
            correct = (pred.argmax(1) == y).clone().detach().sum()

            test_loss += loss.item()
            test_correct += correct.item()

        test_loss /= self.batch_size
        test_correct /= size
        pbar.close()
        print(f"Accuracy: {(100.0*test_correct):>0.1f}%, Loss: {test_loss:>8f} \n")

    def run(self):
        epochs = self.config.epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")
        self.model.save_weights()

