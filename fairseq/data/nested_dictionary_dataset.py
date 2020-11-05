# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch.utils.data.dataloader import default_collate
from fairseq.data import data_utils
from . import FairseqDataset
import numpy as np


def _flatten(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + '.' if prefix is not None else ''
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + '.[' + str(i) + ']'))
    else:
        new_dico = OrderedDict({prefix: dico})
    return new_dico


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split('.')
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith('[') and k.endswith(']'):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    return new_dico


class NestedDictionaryDataset(FairseqDataset):

    def __init__(self, defn, sizes=None, all_sizes=None, padding_idx=None, add_ref_prob=0.0):
        super().__init__()
        self.defn = _flatten(defn)
        self.sizes = [sizes] if not isinstance(sizes, (list, tuple)) else sizes
        self.all_sizes = all_sizes

        first = None
        for v in self.defn.values():
            if not isinstance(v, (FairseqDataset, torch.utils.data.Dataset, )):
                raise ValueError('Expected Dataset but found: {}'.format(v.__class__))
            first = first or v
            if len(v) > 0:
                assert len(v) == len(first), 'dataset lengths must match'

        self._len = len(first)
        self.padding_idx = padding_idx
        self.add_ref_prob = add_ref_prob

    def __getitem__(self, index):
        if self.all_sizes is not None:
            sample = OrderedDict((k, ds[index]) for k, ds in self.defn.items())

            if self.add_ref_prob > 0:
                sample['first_seg_num_tokens'] = (self.all_sizes[0][index] + self.all_sizes[1][index], self.all_sizes[0][index])
            else:
                sample['first_seg_num_tokens'] = (self.all_sizes[0][index], self.all_sizes[0][index])
            if 'parallel_target' in self.defn:
                source_num_tokens = self.all_sizes[0][index]
                target_mask = self.defn['parallel_target'][index]
                sample['parallel_data_mask'] = torch.cat([target_mask.new_full((source_num_tokens+1, ), self.padding_idx),
                                                          target_mask])
            return sample
        else:
            return OrderedDict((k, ds[index]) for k, ds in self.defn.items())

    def __len__(self):
        return self._len

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}
        sample = OrderedDict()

        tripple_inputs = False if self.add_ref_prob == 0. else True
        if 0.0 < self.add_ref_prob < 1.:
            dice_prob = np.random.uniform()
            tripple_inputs = True if dice_prob < self.add_ref_prob else False

        for k, ds in self.defn.items():
            if k == 'target':
                sample[k] = torch.cat([s[k] for s in samples])
            elif 0.0 < self.add_ref_prob < 1. and (k == 'net_input.src_tokens' or k == 'net_input.src_lengths' or k == 'ntokens'):
                sample[k] = ds.collater([s[k][0] for s in samples]) if tripple_inputs else ds.collater([s[k][1] for s in samples])
            else:
                try:
                    sample[k] = ds.collater([s[k] for s in samples])
                except NotImplementedError:
                    sample[k] = default_collate([s[k] for s in samples])

        if 'parallel_data_mask' in samples[0]:
            k = 'parallel_data_mask'
            x = [s[k] for s in samples]
            sample[k] = data_utils.collate_tokens(x, self.padding_idx, left_pad=False)

        if self.all_sizes is not None:
            # "target" is not nested
            full = sample['net_input.src_tokens']
            # print('net_input.src_tokens', full.size(), flush=True)
            full_lengths = sample['net_input.src_lengths']
            first_seg_lengths = default_collate([s['first_seg_num_tokens'][0] if tripple_inputs else s['first_seg_num_tokens'][1] for s in samples])
            # target are stripped, offset labels for target sentences only
            # directly concat as the target in above
            mask = torch.arange(full.size(1)).to(full.device).unsqueeze(0).expand(full.size(0), full.size(1))
            # first_seg_lengths+1 is the start of the target labels;
            # full_lengths-1 is the last token, -2 is the actual last token of target w/o <eos>
            sample['target_mask'] = (mask.ge(first_seg_lengths.unsqueeze(1)+1)) & (mask.le(full_lengths.unsqueeze(1)-2))
            sample['target_lengths'] = default_collate([torch.numel(s['target']) for s in samples])
            assert sample['target_mask'].sum() == sample['target_lengths'].sum()
        unflatten = _unflatten(sample)
        return unflatten

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(s[index] for s in self.sizes)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if len(self.sizes) == 1:
            return self.sizes[0][index]
        else:
            return (s[index] for s in self.sizes)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, 'supports_prefetch', False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.defn.values():
            ds.set_epoch(epoch)
