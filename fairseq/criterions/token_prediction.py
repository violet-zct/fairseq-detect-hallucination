# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils, modules
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('token_prediction')
class TokenPredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, masked_lm_loss_weight):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.masked_lm_loss_weight = masked_lm_loss_weight

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        parser.add_argument('--masked-lm-loss-weight', type=float, default=0.0)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
                hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        if 'parallel_data_mask' in sample:
            parallel_data_mask = sample['parallel_data_mask'].ne(self.padding_idx)
        else:
            parallel_data_mask = None
        logits, extra = model(
            sample['net_input']['src_tokens'],
            features_only=True,
            classification_head_name=self.classification_head_name,
            target_mask=sample['target_mask'],
            parallel_data_mask=parallel_data_mask,
            parallel_data=sample['net_input']['parallel_src_tokens'] if parallel_data_mask is not None else None,
        )

        targets = model.get_targets(sample, [logits]).view(-1)  # K (K=\sum_i B_i)
        sample_size = targets.numel()

        target_lengths = sample['target_lengths']
        assert sum(target_lengths) == sample_size

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets, reduction='sum')

        if parallel_data_mask is not None:
            # compute masked LM loss on the target side
            masked_logits = extra
            parallel_target = sample['parallel_target']
            target_mask = parallel_target.ne(self.padding_idx)
            total_tokens = target_mask.int().sum()
            parallel_target = parallel_target[target_mask]
            masked_prediction_loss = modules.cross_entropy(
                masked_logits.view(-1, masked_logits.size(-1)),
                parallel_target.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )
            masked_lm_loss = masked_prediction_loss / total_tokens
            hallucination_pred_loss = loss / sample_size
            loss = hallucination_pred_loss + self.masked_lm_loss_weight * masked_lm_loss

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size if self.masked_lm_loss_weight <= 0 else 1,
        }
        preds = logits.argmax(dim=1)
        logging_output['ncorrect'] = (preds == targets).sum()

        if parallel_data_mask is not None:
            logging_output['hallucination_pred_loss'] = hallucination_pred_loss.data
            logging_output['masked_lm_loss'] = masked_lm_loss.data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

        if len(logging_outputs) > 0 and 'masked_lm_loss' in logging_outputs[0]:
            masked_lm_loss = sum(log.get('masked_lm_loss', 0) for log in logging_outputs)
            metrics.log_scalar('masked_lm_loss', masked_lm_loss / sample_size / math.log(2) if sample_size > 0 else 0.0,
                               sample_size, round=3)
            hallucination_pred_loss = sum(log.get('hallucination_pred_loss', 0) for log in logging_outputs)
            metrics.log_scalar('hallucination_pred_loss',
                               hallucination_pred_loss / sample_size / math.log(2) if sample_size > 0 else 0.0,
                               sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
