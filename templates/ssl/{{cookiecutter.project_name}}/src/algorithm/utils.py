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
