import torch
from torch.nn.functional import pad
from data.tokenizer import tokenize


def subsequent_mask(size):
    # Mask out subsequent positions. Boolean mask
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class Batch(object):
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = src == pad
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.tgt == pad
            self.ntokens = (self.tgt_y != pad).data.sum()

    def to(self, device):
        self.src = self.src.to(device)
        self.tgt = self.tgt.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        self.tgt_y = self.tgt_y.to(device)
        self.ntokens = self.ntokens.to(device)


class Collator(object):
    def __init__(self, tokenizer, vocab, max_padding):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_padding = max_padding

    def __call__(self, batch):
        bs_id = torch.tensor([0])  # <s> token id
        eos_id = torch.tensor([1])  # </s> token id
        pad_id = 2
        src_list, tgt_list = [], []
        for s in batch:
            _src, _tgt = s["de"], s["en"]
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab["de"](tokenize(_src, self.tokenizer["de"])),
                        dtype=torch.int64,
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        self.vocab["en"](tokenize(_tgt, self.tokenizer["en"])),
                        dtype=torch.int64,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(
                    processed_src,
                    (
                        0,
                        self.max_padding - len(processed_src),
                    ),
                    value=pad_id,
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, self.max_padding - len(processed_tgt)),
                    value=pad_id,
                )
            )

        src = torch.stack(src_list)
        tgt = torch.stack(tgt_list)
        return Batch(src, tgt)
