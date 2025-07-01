class SLTrainer:
    def __init__(self, data, model, config):
        self.model = model
        self.data = data
        self.config = config

    def run_training(self):
        training_loss = 0.0
        for i, batch in enumerate(self.data.train_dataloader):
            loss = self.model.train(batch)
            self.model.update()
            training_loss += loss
            if i % self.config.log_interval == 0:
                print("Logging")
        training_loss /= len(self.data.train_dataloader)
        print(f"Training Loss: {training_loss:>8f}")

    def run_validation(self):
        val_loss = 0.0
        val_accuracy = 0.0
        for _, batch in enumerate(self.data.val_dataloader):
            accuracy, loss = self.model.validate(batch)
            val_loss += loss
            val_accuracy += accuracy
        val_loss /= len(self.data.val_dataloader)
        val_accuracy /= len(self.data.val_dataloader)
        print(
            f"Validation Loss: {val_loss:>8f}, Validation Accuracy: {(100.0 * val_accuracy):>0.1f}"
        )

    def run(self):
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch+1}")
            self.run_training()
            self.run_validation()
