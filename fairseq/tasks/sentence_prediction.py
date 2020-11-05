# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.utils import get_whole_word_mask

logger = logging.getLogger(__name__)


@register_task('sentence_prediction')
class SentencePredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='truncate sequence to max-positions')
        parser.add_argument('--add-prev-output-tokens', action='store_true', default=False,
                            help='add prev_output_tokens to sample, used for encoder-decoder arch')
        parser.add_argument('--add-target-num-tokens', action='store_true', default=False,
                            help='add target num tokens to sample, used for calculating losses')
        parser.add_argument('--input0', type=str, default="input0")
        parser.add_argument('--input1', type=str, default="input1")
        parser.add_argument('--input2', type=str, default=None,
                            help="if not None, use translation loss")
        parser.add_argument('--add-tran-loss', type=int, default=0)

        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.0, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.0, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--add-ref-prob', default=0.0, type=float,
                            help='use reference in training prediction with this probability, '
                                 'if == 1, replace previous use_tripple_inputs, '
                                 'if == 0, only cat source and synthetic target for prediction loss')
        parser.add_argument('--dropout-ref', default=0.0, type=float,
                            help='dropout words in reference with this rate, (for summarization), '
                                 'to avoid easy learning of edit-distance')

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, 'max_positions'):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.add_translation_loss = args.input2 is not None
        if self.add_translation_loss:
            self.mask_idx = self.dictionary.add_symbol('<mask>')

        self.add_ref_prob = args.add_ref_prob

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, args.input0, 'dict.txt'),
            source=True,
        )
        logger.info('[input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, 'label', 'dict.txt'),
                source=False,
            )
            logger.info('[label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict
        return SentencePredictionTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        # input0 is source, input1 is synthetic target, input2 is reference
        input0 = make_dataset(self.args.input0, self.source_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path(type, split))
        input1 = make_dataset(self.args.input1, self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        if self.args.input2 is not None:
            input2 = make_dataset(self.args.input2, self.source_dictionary)

        if self.args.input2 is not None and self.add_ref_prob > 0 and split != 'valid':
            input3 = PrependTokenDataset(input2, self.args.separator_token)
        else:
            input3 = None

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)

            if self.args.input2 is not None and self.add_ref_prob > 0. and split != 'valid':
                src_tokens = ConcatSentencesDataset(input0, input3, input1, add_ref_prob=self.add_ref_prob,
                                                    drop_ref_rate=self.args.dropout_ref,
                                                    pad_idx=self.source_dictionary.pad(),
                                                    eos_idx=self.source_dictionary.eos(),
                                                    bos_idx=self.source_dictionary.bos())
            else:
                src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        if self.args.truncate_sequence:
            src_tokens = TruncateDataset(src_tokens, self.args.max_positions)

        if self.args.input2 is not None and self.args.add_tran_loss:
            # create masked input and targets
            mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
                if self.args.mask_whole_words else None
            ref_dataset, ref_target_dataset = MaskTokensDataset.apply_mask(
                input2,
                self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=mask_whole_words,
            )

            if self.args.separator_token is not None:
                input2 = PrependTokenDataset(ref_dataset, self.args.separator_token)
            parallel_src_tokens = ConcatSentencesDataset(input0, input2)
            if self.args.truncate_sequence:
                parallel_src_tokens = TruncateDataset(parallel_src_tokens, self.args.max_positions)

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        if self.args.input2 is not None and self.args.add_tran_loss:
            dataset['net_input']['parallel_src_tokens'] = RightPadDataset(parallel_src_tokens,
                                                                          pad_idx=self.source_dictionary.pad(),
                                                                          )

        if self.args.add_prev_output_tokens:
            prev_tokens_dataset = RightPadDataset(
                RollDataset(src_tokens, 1),
                pad_idx=self.dictionary.pad(),
            )
            dataset['net_input'].update(
                prev_output_tokens=prev_tokens_dataset,
            )

        if not self.args.regression_target:
            label_dataset = make_dataset('label', self.label_dictionary)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.label_dictionary.eos(),
                        ),
                        offset=-self.label_dictionary.nspecial,
                    )
                )
            if self.args.input2 is not None and self.args.add_tran_loss:
                # used as translation target when calculating loss
                dataset.update(parallel_target=RightPadDataset(ref_target_dataset,
                                                               pad_idx=self.source_dictionary.pad(),
                                                               ))
        else:
            label_path = "{0}.label".format(get_path('label', split))
            if os.path.exists(label_path):
                def parse_regression_target(i, line):
                    values = line.split()
                    assert len(values) == self.args.num_classes, \
                        f'expected num_classes={self.args.num_classes} regression target values on line {i}, found: "{line}"'
                    return [float(x) for x in values]

                dataset.update(
                    target=RawLabelDataset([
                        parse_regression_target(i, line.strip()) for i, line in enumerate(open(label_path).readlines())
                    ])
                )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
            all_sizes=src_tokens.all_sizes if self.args.add_target_num_tokens else None,
            padding_idx=self.source_dictionary.pad(),
            add_ref_prob=self.add_ref_prob if split != 'valid' else 0.,
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_head(
            getattr(args, 'classification_head_name', 'sentence_classification_head'),
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad())
