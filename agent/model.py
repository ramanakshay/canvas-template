import torch
from torch import nn

from agent.network import NeuralNetwork

class Model(object):
    def __init__(self, config):
        self.config = config
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        
        self.network = NeuralNetwork().to(self.device)
        
        self.learning_rate = self.config.learning_rate
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.grad_enabled = False
    
    def __repr__(self):
        return str(self.network)
    
    def enable_grad(self, mode):
        self.grad_enabled = mode
            
    def save(self):
        torch.save(self.network.state_dict(), "model.pth")

    def reset(self):
        self.optimizer.zero_grad()

    def update(self):
        self.optimizer.step()

    def get_prediction(self, X):
        with torch.set_grad_enabled(self.grad_enabled):
            X = X.to(self.device)
            pred = self.network(X)
            return pred
        
    