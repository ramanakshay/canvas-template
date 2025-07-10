import torch
from torch import nn
from model.transformer import Transformer
from torch.optim.lr_scheduler import LambdaLR
from model.loss import SimpleLossCompute, LabelSmoothing


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


class TransformerModel:
    def __init__(self, src_vocab, tgt_vocab, config, device):
        self.config = config
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.device = device
        self.transformer = Transformer(
            src_vocab,
            tgt_vocab,
            d_model=self.config.network.d_model,
            nhead=self.config.network.h,
            num_encoder_layers=self.config.network.N,
            num_decoder_layers=self.config.network.N,
            dim_feedforward=self.config.network.d_ff,
            dropout=self.config.network.dropout,
        ).to(self.device)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        criterion = LabelSmoothing(
            tgt_vocab, padding_idx=2, smoothing=self.config.loss.smoothing
        )
        self.loss = SimpleLossCompute(self.transformer.generator, criterion)
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(),
            lr=self.config.optimizer.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda step: rate(
                step,
                model_size=self.config.network.d_model,
                factor=1.0,
                warmup=self.config.optimizer.warmup,
            ),
        )

    def set_mode(self, is_training=True):
        self.transformer.train(is_training)

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def predict(self, src, tgt, src_mask, tgt_mask):
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        src_mask = src_mask.to(self.device)
        tgt_mask = tgt_mask.to(self.device)
        return self.transformer(src, tgt, src_mask, tgt_mask)

    def generate(self, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        memory = self.transformer.encode(src, src_mask)
        ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
        for i in range(max_len - 1):
            out = self.transformer.decode(memory, ys, src_mask)
            prob = self.transformer.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat(
                [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
            )
        return ys

    def train(self, batch):
        batch.to(self.device)
        pred = self.transformer(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = self.loss(pred, batch.tgt_y, batch.ntokens)
        loss_node.backward()
        return loss

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def scheduler_step(self):
        self.scheduler.step()
