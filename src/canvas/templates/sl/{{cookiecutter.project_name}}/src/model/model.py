import torch
from torch import nn, optim
from model.network import MLP


class ClassifierModel:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.network = MLP(
            self.config.input_dim, self.config.output_dim, self.config.hidden_dims
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.loss = nn.CrossEntropyLoss()

    def set_mode(self, is_training=True):
        self.network.train(is_training)

    def predict(self, X):
        pred = self.network(X)
        return pred

    def train(self, batch):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        pred = self.network(X)
        loss = self.loss(pred, y)
        loss.backward()
        return loss.item()

    def update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def validate(self, batch):
        X, y = batch
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            pred = self.network(X)
        loss = self.loss(pred, y).item()
        accuracy = (pred.argmax(1) == y).clone().detach().sum().item()
        return loss, accuracy
