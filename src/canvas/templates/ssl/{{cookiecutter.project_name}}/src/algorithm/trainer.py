import time


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class SSLTrainer:
    def __init__(self, data, model, config):
        self.data = data
        self.model = model
        self.config = config
        self.train_state = TrainState()

    def run_training(self):
        self.model.set_mode(is_training=True)
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        train_state = self.train_state

        for i, batch in enumerate(self.data.train_dataloader):
            loss = self.model.train(batch)
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % self.config.accum_interval == 0:
                self.model.optimizer_step()
                self.model.scheduler_step()
                n_accum += 1
                train_state.accum_step += 1
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % self.config.log_interval == 0:
                lr = self.model.get_lr()
                elapsed = time.time() - start
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                start = time.time()
                tokens = 0
        training_loss = total_loss / total_tokens
        print(f"Epoch Loss: {training_loss}")

    def run(self):
        for epoch in range(self.config.epochs):
            self.run_training()
