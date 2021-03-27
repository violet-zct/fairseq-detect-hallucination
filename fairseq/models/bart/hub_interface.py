# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from fairseq import utils
from fairseq.data import encoders
import math
from fairseq.data.encoders.utils import get_whole_word_mask
logger = logging.getLogger(__name__)


def add_insertion_noise(tokens, p, mask_idx, vocab_length, random_ratio=0.0):
    if p == 0.0:
        return tokens

    num_tokens = len(tokens)
    n = int(math.ceil(num_tokens * p))

    noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
    noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
    noise_mask[noise_indices] = 1
    result = torch.LongTensor(n + len(tokens)).fill_(-1)

    num_random = int(math.ceil(n * random_ratio))
    result[noise_indices[num_random:]] = mask_idx
    result[noise_indices[:num_random]] = torch.randint(low=1, high=vocab_length, size=(num_random,))

    result[~noise_mask] = tokens

    assert (result >= 0).all()
    return result


def apply_mask_noise(source, mask_idx, vocab_length, mask_length="span-poisson", mask_ratio=0.3,
                     random_ratio=0.1, poisson_lambda=3.5, replace_length=1, insert_ratio=0.2,
                     mask_whole_word=None):
    def _word_starts(source):
        if mask_whole_word is not None:
            is_word_start = mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    if mask_length == "span-poisson":
        _lambda = poisson_lambda
        lambda_to_the_k = 1
        e_to_the_minus_lambda = math.exp(-_lambda)
        k_factorial = 1
        ps = []
        for k in range(0, 128):
            ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
            lambda_to_the_k *= _lambda
            k_factorial *= (k + 1)
            if ps[-1] < 0.0000001:
                break
        ps = torch.FloatTensor(ps)
        mask_span_distribution = torch.distributions.Categorical(ps)
    else:
        mask_span_distribution = None

    is_word_start = _word_starts(source)
    num_to_mask = int(math.ceil(is_word_start.float().sum() * mask_ratio))
    num_inserts = 0
    if num_to_mask == 0:
        return source

    if mask_span_distribution is not None:
        lengths = mask_span_distribution.sample(sample_shape=(num_to_mask,))

        # Make sure we have enough to mask
        cum_length = torch.cumsum(lengths, 0)
        while cum_length[-1] < num_to_mask:
            lengths = torch.cat([lengths, mask_span_distribution.sample(sample_shape=(num_to_mask,))], dim=0)
            cum_length = torch.cumsum(lengths, 0)

        # Trim to masking budget
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i + 1
        lengths = lengths[:num_to_mask]

        # Handle 0-length mask (inserts) separately
        lengths = lengths[lengths > 0]
        num_inserts = num_to_mask - lengths.size(0)
        num_to_mask -= num_inserts
        if num_to_mask == 0:
            return add_insertion_noise(source, num_inserts / source.size(0), mask_idx, vocab_length)

        assert (lengths > 0).all()
    else:
        lengths = torch.ones((num_to_mask,)).long()
    assert is_word_start[-1] == 0
    word_starts = is_word_start.nonzero()
    indices = word_starts[torch.randperm(word_starts.size(0))[:num_to_mask]].squeeze(1)
    mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_ratio

    source_length = source.size(0)
    assert source_length - 1 not in indices
    to_keep = torch.ones(source_length, dtype=torch.bool)
    is_word_start[-1] = 255  # acts as a long length, so spans don't go over the end of doc
    if replace_length == 0:
        to_keep[indices] = 0
    else:
        # keep index, but replace it with [MASK]
        source[indices] = mask_idx
        source[indices[mask_random]] = torch.randint(1, vocab_length, size=(mask_random.sum(),))

    if mask_span_distribution is not None:
        assert len(lengths.size()) == 1
        assert lengths.size() == indices.size()
        lengths -= 1
        while indices.size(0) > 0:
            assert lengths.size() == indices.size()
            lengths -= is_word_start[indices + 1].long()
            uncompleted = lengths >= 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            lengths = lengths[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = mask_idx
                source[indices[mask_random]] = torch.randint(1, vocab_length, size=(mask_random.sum(),))
    else:
        # A bit faster when all lengths are 1
        while indices.size(0) > 0:
            uncompleted = is_word_start[indices + 1] == 0
            indices = indices[uncompleted] + 1
            mask_random = mask_random[uncompleted]
            if replace_length != -1:
                # delete token
                to_keep[indices] = 0
            else:
                # keep index, but replace it with [MASK]
                source[indices] = mask_idx
                source[indices[mask_random]] = torch.randint(1, vocab_length, size=(mask_random.sum(),))

            assert source_length - 1 not in indices

    source = source[to_keep]

    if num_inserts > 0:
        source = add_insertion_noise(source, num_inserts / source.size(0), mask_idx, vocab_length)

    if insert_ratio > 0:
        source = add_insertion_noise(source, insert_ratio, mask_idx, vocab_length)
    return source


class BARTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/BART
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model
        self.bpe = encoders.build_bpe(args)

        self.max_positions = min(utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
        ))

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))
        self.noise = False

    @property
    def device(self):
        return self._float_tensor.device

    def sample_noise(self):
        pass

    def encode(self, sentence: str, *addl_sentences, no_separator=True) -> (torch.LongTensor, torch.LongTensor):
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`).

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> bart.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> bart.encode(' world').tolist()
            [0, 232, 2]
            >>> bart.encode('world').tolist()
            [0, 8331, 2]
        """
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions - 2:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions - 2])
        bpe_sentence = '<s> ' + tokens + ' </s>'
        for s in addl_sentences:
            bpe_sentence += (' </s>' if not no_separator else '')
            bpe_sentence += ' ' + self.bpe.encode(s) + ' </s>'
        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False).long()

        if self.noise:
            insert_prob = self.noise_params['insert_ratio']
            if "high_mask_prob" in self.noise_params and self.noise_params["high_mask_prob"] > 0:
                mask_prob = np.random.uniform(self.noise_params['low_mask_prob'], self.noise_params['high_mask_prob'])
                random_prob = np.random.uniform(self.noise_params['low_random_prob'],
                                                self.noise_params['high_random_prob'])
                if self.noise_params['random_word_span']:
                    random_word_span = 'none' if np.random.uniform(high=1) > 0.5 else "span-poisson"
                else:
                    random_word_span = self.noise_params['mask_length']
            else:
                mask_prob = self.noise_params['mask_ratio']
                random_prob = self.noise_params['random_ratio']
                random_word_span = self.noise_params['mask_length']

            noised_tokens = apply_mask_noise(tokens.clone(), self.mask_idx, self.vocab_length,
                                         random_word_span, mask_prob,
                                         random_prob, self.noise_params['poisson_lambda'],
                                         self.noise_params['replace_length'], insert_prob,
                                         self.noise_params['mask_whole_word'])
        else:
            noised_tokens = tokens
        # print(tokens.size(), noised_tokens.size())
        return (tokens, noised_tokens)

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def sample(self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs) -> (List[str], List[str]):
        kwargs["lenpen"] = 1.5

        pair_input = [self.encode(sentence) for sentence in sentences]
        intact_input = [inpt[0] for inpt in pair_input]
        input = [inpt[1] for inpt in pair_input]
        if verbose:
            src_str_with_unk = self.string(intact_input)
            logger.info('S\t{}'.format(src_str_with_unk))

        hypos = self.generate(input, beam, verbose, **kwargs)
        noised_src = [self.decode(s) for s in input]
        decoded = [self.decode(x['tokens']) for x in hypos]
        if verbose:
            for src, nsrc, hypo in zip(intact_input, noised_src, decoded):
                logger.info('Noised-S\t{}'.format(nsrc))
                logger.info('T\t{}'.format(hypo))
        return decoded, noised_src

    def generate(self, tokens: List[torch.LongTensor], beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=sample['net_input']['src_tokens'].new_zeros((len(tokens), 1)).fill_(self.task.source_dictionary.bos()),
        )

        # if verbose:
        #     src_str_with_unk = self.string(tokens)
        #     logger.info('S\t{}'.format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(dim=1)- 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    def predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def load_noise_hyperparam(self, noise_params):
        self.noise = True
        self.noise_params = noise_params
        self.seed_noise_params = noise_params
        self.mask_idx = self.task.mask_idx
        self.vocab_length = len(self.task.source_dictionary)
        if self.noise_params['mask_whole_word']:
            self.noise_params['mask_whole_word'] = get_whole_word_mask(self.args, self.task.source_dictionary)
        else:
            self.noise_params['mask_whole_word'] = None
        self.seed = noise_params['seed']
        np.random.seed(self.seed)
        utils.set_torch_seed(self.seed)

    def set_noise(self, mask, random):
        self.noise_params['mask_ratio'] = mask
        self.noise_params['random_ratio'] = random

    def set_high_noise(self):
        self.noise_params['mask_ratio'] = 0.8
        self.noise_params['insert_ratio'] = 0.2
        self.noise_params['random_ratio'] = 0.4

        self.noise_params['mask_length'] = 'none'

    def restore_noise(self):
        for k, v in self.seed_noise_params.items():
            self.noise_params[k] = v