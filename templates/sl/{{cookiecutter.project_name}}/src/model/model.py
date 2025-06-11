import torch
from torch import nn, optim
from model.network import MLP

class ClassifierModel:
    def __init__(self, config, device):
        self.config = config.model
        self.device = device

        self.network = MLP(self.config.input_dim, self.config.hidden_dim,
                           self.config.output_dim).to(self.device)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=self.config.learning_rate,
                                    weight_decay=self.config.weight_decay)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def predict(self, image):
        pred = self.network(image)
        return pred

    def learn(self, pred, label):
        loss = self.loss(pred, label)
        loss.backward()
        return loss

    def update(self, pred, label):
        self.optimizer.step()
        self.optimizer.zero_grad()


