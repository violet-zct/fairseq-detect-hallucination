# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from loss_dropper import LossDropper


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('drop_loss_label_smoothed_cross_entropy')
class Drop_Loss_LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, word_filter, drop_c):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.word_filter = word_filter
        self.drop_c = drop_c
        if drop_c > 0:
            self.dropper = LossDropper(dropc=drop_c)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--word-filter', default=0, type=int, help="set to 1 to turn on our word filter loss")
        parser.add_argument('--drop-c', default=0, type=float, help="truncation loss from (Kang and Hashimoto, 2020)")
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.training and "target_labels" in sample:
            sample['net_input']["target_label_mask"] = 1. - sample["target_labels"]
        net_output = model(**sample['net_input'])
        if self.training and "target_labels" in sample and self.word_filter:
            loss, nll_loss = self.compute_token_loss(model, net_output, sample, sample['target_labels'])
            sample_size = sample['target'].size(0) if self.sentence_avg else (1 - sample['target_labels']).sum().item()
        elif self.training and self.drop_c > 0:
            loss, nll_loss, pad_mask = self.compute_drop_loss(model, net_output, sample)
            mask = self.dropper(loss)  # The dropper returns a mask of 0s where data should be dropped.
            loss *= mask
            loss = loss.sum()
            sample_size = mask.sum()
            # sample_size = mask.sum() if self.sentence_avg else (pad_mask*mask).sum()
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_token_loss(self, model, net_output, sample, target_weights):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # mask = (sample['target'] != self.padding_idx).float()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
        )
        token_loss = loss.reshape_as(sample['target']) * (1.0 - target_weights)
        nll_loss = nll_loss.reshape_as(sample['target']) * (1.0 - target_weights)
        return token_loss.sum(), nll_loss.sum()

    def compute_drop_loss(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # mask = (sample['target'] != self.padding_idx).float()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
        )
        mask = (sample['target'] != self.padding_idx).float()
        loss = loss.reshape_as(sample['target'])
        token_loss = (loss * mask).sum(1) / mask.sum(1)
        nll_loss = nll_loss.reshape_as(sample['target'])
        return token_loss, nll_loss.sum(), mask.sum(1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
