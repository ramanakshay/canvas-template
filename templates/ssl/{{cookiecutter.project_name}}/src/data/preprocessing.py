import torch
from torch.nn.functional import pad
from data.tokenizer import tokenize


class Collator(object):
    def __init__(self, tokenizer, vocab, device, max_padding):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        self.max_padding = max_padding

    def __call__(self, batch):
        bs_id = torch.tensor([0], device=self.device)  # <s> token id
        eos_id = torch.tensor([1], device=self.device)  # </s> token id
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
                        device=self.device,
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
                        device=self.device,
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
        return src, tgt
