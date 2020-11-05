# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import FairseqDataset


class ConcatSentencesDataset(FairseqDataset):

    def __init__(self, *datasets, add_ref_prob=0.0, drop_ref_rate=0.0, pad_idx=0, eos_idx=2, bos_idx=3):
        super().__init__()
        self.datasets = datasets
        assert all(len(ds) == len(datasets[0]) for ds in datasets), \
            'datasets must have the same length'
        self.add_ref_prob = add_ref_prob
        self.drop_ref_rate = drop_ref_rate
        self.padding_idx = pad_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx

    def mask_words(self, src_tokens):
        src_masks = src_tokens.eq(self.padding_idx) | src_tokens.eq(self.eos_idx) | src_tokens.eq(self.bos_idx)
        full_length = len(src_tokens)
        if full_length <= 2:
            return src_tokens
        mask_lengths = int(full_length * self.drop_ref_rate)
        mask = torch.arange(full_length).to(src_tokens.device).ge(mask_lengths)
        mask = mask.long()
        scores = src_tokens.clone().float().uniform_()
        scores.masked_fill_(src_masks, -1)
        sorted_values, sorted_idx = torch.sort(scores, descending=True)
        mask = mask.scatter(0, sorted_idx, mask)  # 0 are dropped words
        src_tokens[(1 - mask).bool()] = self.padding_idx
        return src_tokens

    def __getitem__(self, index):
        # todo: ad-hoc change; warning in the future
        if self.add_ref_prob == 0. or self.add_ref_prob == 1.:
            if self.add_ref_prob == 1. and self.drop_ref_rate > 0:
                ref = self.mask_words(self.datasets[1][index])
                return torch.cat([self.datasets[0][index], ref, self.datasets[2][index]])
            else:
                # default call
                return torch.cat([ds[index] for ds in self.datasets])
        else:
            return (torch.cat([ds[index] for ds in self.datasets]),
                    torch.cat([self.datasets[0][index], self.datasets[-1][index]]))

    def __len__(self):
        return len(self.datasets[0])

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        return sum(ds.sizes for ds in self.datasets)

    def num_tokens(self, index):
        return sum(ds.num_tokens(index) for ds in self.datasets)

    @property
    def all_sizes(self):
        return [ds.sizes for ds in self.datasets]

    def size(self, index):
        return sum(ds.size(index) for ds in self.datasets)

    def ordered_indices(self):
        return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(
            getattr(ds, 'supports_prefetch', False) for ds in self.datasets
        )

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, 'set_epoch'):
                ds.set_epoch(epoch)
