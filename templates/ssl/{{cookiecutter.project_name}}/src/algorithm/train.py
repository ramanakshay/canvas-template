import time
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from algorithm.utils import Batch
from algorithm.loss import SimpleLossCompute, LabelSmoothing


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class SSLTrainer:
    def __init__(self, data, model, config, device):
        self.data = data
        self.model = model
        self.config = config.algorithm
        self.dataloaders = self.data.get_dataloaders()
        self.device = device

        self.accum_iter = self.config.accum_iter
        self.train_state = TrainState()

    def run_epoch(self, mode):
        self.model.train()
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        n_accum = 0
        train_state = self.train_state

        for i, b in enumerate(self.dataloaders[mode]):
            batch = Batch(b[0], b[1], pad=2)
            batch.to(device)
            pred = self.model.predict(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            target, norm = batch.tgt_y, batch.ntokens

            if mode == 'train':
                loss, loss_node = self.model.learn(pred, target, norm)
                train_state.step += 1
                train_state.samples += batch.src.shape[0]
                train_state.tokens += batch.ntokens
                if i % self.accum_iter == 0:
                    self.model.update()
                    n_accum += 1
                    train_state.accum_step += 1

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % 40 == 1 and (mode == "train"):
                lr = self.model.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(
                    ( "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                     + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e" )
                    % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
                )
                start = time.time()
                tokens = 0

            del loss
            del loss_node
        return total_loss / total_tokens

    def run(self):
        for epoch in range(self.config.epochs):
            self.model.train()
            self.run_epoch(mode='train')
            self.model.eval()
            self.run_epoch(mode='valid')
