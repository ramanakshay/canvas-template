import torch
from tqdm import tqdm


class SLTrainer:
    def __init__(self, data, model, config, device):
        self.model = model
        self.data = data
        self.config = config.algorithm
        self.device = device

    def train(self):
        self.model.train()
        pbar = tqdm(self.data.train_dataloader)
        pbar.set_description('Train')
        for i, (X, y) in enumerate(pbar):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model.predict(X)
            loss = self.model.learn(pred, y)
            self.model.update()
            if i % 40 == 1: # update every 40 steps
                pbar.set_postfix(loss=loss.item())
        pbar.close()

    def test(self):
        size = len(self.data.test_dataset)
        batch_size = self.data.config.batch_size
        self.model.eval()
        pbar = tqdm(self.data.test_dataloader)
        pbar.set_description('Test')
        test_loss, test_correct = 0.0, 0.0
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)
            with torch.no_grad():
                pred = self.model.predict(X)
                loss = self.model.loss(pred, y)
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
            self.train()
            self.test()
