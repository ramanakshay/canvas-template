import torch
from torch import nn, optim
from tqdm import tqdm


class Trainer:
    def __init__(self, data, model, config):
        self.model = model
        self.data = data
        self.dataloaders = data.get_dataloaders()
        self.config = config.algorithm

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.network.parameters(),
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)
        self.scheduler = None

    def run_train_epoch(self):
        self.model.enable_grad(True)
        pbar = tqdm(self.dataloaders['train'])
        pbar.set_description('Train')
        for i, (X, y) in enumerate(pbar):
            self.optimizer.zero_grad()
            y = y.to(self.model.device)
            pred = self.model.predict(X)
            loss = self.loss_function(pred, y)
            loss.backward()
            self.optimizer.step()
            if i % 40 == 1: # update every 40 steps
                pbar.set_postfix(loss=loss.item())
        pbar.close()
        if self.scheduler is not None: self.scheduler.step()

    def run_test_epoch(self):
        size = len(self.dataloaders['test'].dataset)
        batch_size = self.dataloaders['test'].batch_size
        self.model.enable_grad(False)
        pbar = tqdm(self.dataloaders['test'])
        pbar.set_description('Test')
        test_loss, test_correct = 0.0, 0.0
        for X, y in pbar:
            y = y.to(self.model.device)
            pred = self.model.predict(X)
            loss = self.loss_function(pred, y)
            correct = (pred.argmax(1) == y).clone().detach().sum()
            test_loss += loss.item()
            test_correct += correct.item()
        test_loss /= batch_size
        test_correct /= size
        pbar.close()
        print(f"Accuracy: {(100.0*test_correct):>0.1f}%, Loss: {test_loss:>8f} \n")

    def run(self):
        epochs = self.config.epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.run_train_epoch()
            self.run_test_epoch()
        self.model.save_weights()

