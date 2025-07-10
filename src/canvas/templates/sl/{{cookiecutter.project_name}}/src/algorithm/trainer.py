import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


class SLTrainer:
    def __init__(self, data, model, config):
        self.model = model
        self.data = data
        self.config = config

    def run_validation(self):
        self.model.eval_mode()
        val_loss = 0.0
        val_accuracy = 0.0
        steps_per_epoch = len(self.data.val_dataloader)

        for _, batch in enumerate(self.data.val_dataloader):
            loss, accuracy = self.model.validate(batch)
            val_loss += loss
            val_accuracy += accuracy

        val_loss /= steps_per_epoch
        val_accuracy /= steps_per_epoch
        metrics = {"loss": val_loss, "accuracy": val_accuracy}
        return metrics

    def run_training(self, epoch, progress_bar):
        self.model.train_mode()
        steps_per_epoch = len(self.data.train_dataloader)
        training_loss = 0.0
        for step, batch in enumerate(self.data.train_dataloader, start=1):
            loss = self.model.train(
                batch, grad_accum_steps=self.config.grad_accum_steps
            )
            training_loss += loss

            if step % self.config.grad_accum_steps == 0:
                self.model.optimizer_step()
                progress_bar.update(1)
                current_epoch_fraction = epoch - 1 + (step / steps_per_epoch)
                progress_bar.set_postfix(
                    {"epoch": f"{current_epoch_fraction:.2f}/{self.config.epochs:.2f}"}
                )

            if step % self.config.log_interval == 0:
                logger.debug(
                    f"Epoch {epoch} | Step {step:04d}/{steps_per_epoch} | Training Loss: {loss:.4f}"
                )

        training_loss /= steps_per_epoch
        metrics = {"loss": training_loss}
        return metrics

    def run(self):
        logger.info("Starting training...")
        total_steps = (
            self.config.epochs
            * len(self.data.train_dataloader)
            // self.config.grad_accum_steps
        )
        effective_batch_size = (
            self.data.train_dataloader.batch_size * self.config.grad_accum_steps
        )
        logger.info(f"Num Epochs: {self.config.epochs}")
        logger.info(f"Gradient Accumulation: {self.config.grad_accum_steps}")
        logger.info(f"Effective Batch Size: {effective_batch_size}")
        logger.info(f"Total Steps: {total_steps}")

        with (
            logging_redirect_tqdm(),
            tqdm(
                total=total_steps, desc="Training Progress", unit="step"
            ) as progress_bar,
        ):
            for epoch in range(1, self.config.epochs + 1):
                logger.info(f"Epoch {epoch}")
                train_metrics = self.run_training(epoch, progress_bar)
                train_loss = train_metrics["loss"]
                logger.info(f"Training Loss: {train_loss:>8f}")
                val_metrics = self.run_validation()
                val_loss, val_accuracy = val_metrics["loss"], val_metrics["accuracy"]
                logger.info(
                    f"Validation Loss: {val_loss:>8f}, Validation Accuracy: {(100.0 * val_accuracy):>0.1f}%"
                )
                self.model.scheduler_step(val_loss)

        self.model.save()
        logger.info("Training completed.")
