import torch
from torch.utils.data import DataLoader
from torch import nn


class Algorithm(object):
    def __init__(self, dataset, model, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        self.batch_size = self.config.batch_size
        self.train_dataloader = DataLoader(self.dataset["train"], batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.dataset["test"], batch_size=self.batch_size)

        self.loss_function = nn.CrossEntropyLoss()
    
    def train(self):
        size = len(self.train_dataloader.dataset)
        self.model.enable_grad(True)
        self.model.reset()
        for batch, (X, y) in enumerate(self.train_dataloader):
            y = y.to(self.model.device)
            pred = self.model.get_prediction(X)
            loss = self.loss_function(pred, y)
            loss.backward()
            self.model.update()
            self.model.reset()

            if batch % 100 == 0:
                train_loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")
                
    def test(self):
        size = len(self.test_dataloader.dataset)
        self.model.enable_grad(False)
        test_loss, test_correct = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                y = y.to(self.model.device)
                pred = self.model.get_prediction(X)
                loss = self.loss_function(pred, y)
                correct = (pred.argmax(1) == y).type(torch.float).sum()

                test_loss += loss.item()
                test_correct += correct.item()
        
        test_loss /= self.batch_size
        test_correct /= size
        
        print(f"Test Error: \n Accuracy: {(100.0*test_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def run(self):
        epochs = self.config.epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self.train()
            self.test()
        print("Done!")
        
    