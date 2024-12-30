import torch
from torch import nn

from .network import MLP

class Classifier(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = (
            self.config.device if self.config.device != "auto"
            else "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.network = MLP(self.config.input_dim, self.config.hidden_dim,
                           self.config.output_dim).to(self.device)
        self.grad_enabled = False

        if self.config.from_pretrained:
            self.load_weights()

    def __repr__(self):
        return str(self.network)

    def enable_grad(self, mode):
        self.grad_enabled = mode

    def load_weights(self):
        self.network.load_state_dict(torch.load(f'{self.config.weights_path}model.pth', weights_only=True))

    def save_weights(self):
        torch.save(self.network.state_dict(), f'{self.config.weights_path}model.pth')

    def predict(self, image):
        with torch.set_grad_enabled(self.grad_enabled):
            image = image.to(self.device)
            pred = self.network(image)
            return pred

