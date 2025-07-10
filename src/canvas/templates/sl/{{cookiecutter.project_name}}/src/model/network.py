from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, images):
        x = self.flatten(images)
        preds = self.network(x)
        return preds
