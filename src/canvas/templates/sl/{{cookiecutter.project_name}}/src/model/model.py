from model.network import MLP


class ClassifierModel:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.network = MLP(
            self.config.input_dim, self.config.hidden_dim, self.config.output_dim
        ).to(self.device)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def predict(self, image):
        pred = self.network(image)
        return pred
