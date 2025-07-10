import torch
from torch import nn, optim
from model.network import MLP
import logging
import os

logger = logging.getLogger(__name__)


class ClassifierModel:
    def __init__(self, config, device):
        logger.info("Creating model...")
        self.config = config
        self.device = device

        self.network = MLP(
            self.config.network.input_dim,
            self.config.network.output_dim,
            self.config.network.hidden_dims,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.loss = nn.CrossEntropyLoss()

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler.factor,
            patience=self.config.scheduler.patience,
        )

        if self.config.load_from_checkpoint:
            self.load()

        num_trainable_params = sum(
            p.numel() for p in self.network.parameters() if p.requires_grad
        )
        num_total_params = sum(p.numel() for p in self.network.parameters())
        logger.info(
            f"Trainable Parameters: {num_trainable_params:,} of {num_total_params:,} ({num_trainable_params / num_total_params:.2%})"
        )
        logger.info("Model created.")

    def train_mode(self):
        self.network.train(True)

    def eval_mode(self):
        self.network.train(False)

    def predict(self, image):
        image = image.to(self.device)
        with torch.no_grad():
            pred = self.network(image)
        return pred

    def train(self, batch, grad_accum_steps=1):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        preds = self.network(images)
        loss = self.loss(preds, labels) / grad_accum_steps
        loss.backward()
        return loss.item() * grad_accum_steps

    def validate(self, batch):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        with torch.no_grad():
            preds = self.network(images)
        loss = self.loss(preds, labels).item()
        accuracy = (
            preds.argmax(1) == labels
        ).clone().detach().sum().item() / labels.size(0)
        return loss, accuracy

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def scheduler_step(self, val_loss):
        self.scheduler.step(val_loss)

    def save(self):
        save_path = self.config.checkpoint_path
        logger.info(f"Saving model to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.network.state_dict(), save_path)

    def load(self):
        load_path = self.config.checkpoint_path
        if os.path.exists(load_path):
            logger.info(f"Loading model from {load_path}")
            self.network.load_state_dict(
                torch.load(load_path, map_location=self.device)
            )
        else:
            logger.error(f"Checkpoint path not found: {load_path}")
            raise FileNotFoundError(f"Checkpoint path not found: {load_path}")
