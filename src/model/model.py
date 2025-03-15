import torch
from torch import nn
from model.network import MLP

class Classifier:
    def __init__(self, config, device):
        self.config = config.model
        self.device = device

        self.network = MLP(self.config.input_dim, self.config.hidden_dim,
                           self.config.output_dim).to(self.device)

        if self.config.from_pretrained:
            self.load_weights()


    def __repr__(self):
        return str(self.network)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def predict(self, image):
        pred = self.network(image)
        return pred

