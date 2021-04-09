# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import encoders
from fairseq.models.bart.hub_interface import apply_mask_noise
from typing import List
from fairseq.data.encoders.utils import get_whole_word_mask


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))
        self.noise = False

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, sentence: str, *addl_sentences, no_separator=False, raw=True):
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        def assign_lengths(max_length_allowed, addl_lengths_list):
            # priority from the last
            new_addl_lengths = [0] * len(addl_lengths_list)
            remain_length = max_length_allowed
            for idx, length in enumerate(addl_lengths_list[::-1]):
                if length >= max_length_allowed:
                    new_addl_lengths[len(addl_lengths_list)-1-idx] = remain_length // 2
                    remain_length -= (remain_length // 2)
                else:
                    new_addl_lengths[len(addl_lengths_list)-1-idx] = length
                    remain_length -= length
            return remain_length, new_addl_lengths

        second_seg_bpe = None
        if raw:
            first_seg_bpe = self.bpe.encode(sentence)
            appended_seg_bpe = [self.bpe.encode(s) for ii, s in enumerate(addl_sentences)]
            length_appended_bpe = [len(seg.split()) for seg in appended_seg_bpe]
            special_tokens = 6 if len(appended_seg_bpe) == 2 else 4
            total_length = self.model.max_positions() - special_tokens

            allowed_first_seg_length, addl_lengths = assign_lengths(total_length, length_appended_bpe)
            appended_seg_bpe = [" ".join(bpe.split()[:ll]) for ll, bpe in zip(addl_lengths, appended_seg_bpe)]
            first_seg_bpe = " ".join(first_seg_bpe.split()[:allowed_first_seg_length])

            first_seg_length = len(first_seg_bpe.split()) + 3
            bpe_sentence = '<s> ' + first_seg_bpe + ' </s>'
            for ii, second_seg_bpe in enumerate(appended_seg_bpe):
                if len(addl_sentences) == 2 and ii == 0:
                    first_seg_length += (len(second_seg_bpe.split()) + 2)
                bpe_sentence += (' </s>' if not no_separator else '')
                bpe_sentence += ' ' + second_seg_bpe + ' </s>'
        else:
            bpe_sentence = '<s> ' + sentence + ' </s>'
            for s in addl_sentences:
                bpe_sentence += (' </s>' if not no_separator else '')
                bpe_sentence += ' ' + s + ' </s>'

        tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False).long()
        if self.noise:
            # replace_length = -1: don't delete, maks_length / mask_ratio => keep the length
            noised_tokens = apply_mask_noise(tokens.clone(), self.mask_idx, self.vocab_length,
                                             self.noise_params['mask_length'], self.noise_params['mask_ratio'],
                                             self.noise_params['random_ratio'], self.noise_params['poisson_lambda'],
                                             self.noise_params['replace_length'], self.noise_params['insert_ratio'])
            # print(tokens.size(), noised_tokens.size())
            tokens = noised_tokens

        if raw:
            return tokens, first_seg_length, second_seg_bpe
        else:
            return tokens

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

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def extract_features(self, tokens: torch.LongTensor, return_all_hiddens: bool = False) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > self.model.max_positions():
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        features, extra = self.model(
            tokens.to(device=self.device),
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
        features = self.extract_features(tokens.to(device=self.device))
        logits = self.model.classification_heads[head](features)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def extract_features_aligned_to_words(self, sentence: str, return_all_hiddens: bool = False) -> torch.Tensor:
        """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
        from fairseq.models.roberta import alignment_utils
        from spacy.tokens import Doc

        nlp = alignment_utils.spacy_nlp()
        tokenizer = alignment_utils.spacy_tokenizer()

        # tokenize both with GPT-2 BPE and spaCy
        bpe_toks, _, _ = self.encode(sentence)
        spacy_toks = tokenizer(sentence)
        spacy_toks_ws = [t.text_with_ws for t in tokenizer(sentence)]
        alignment = alignment_utils.align_bpe_to_words(self, bpe_toks, spacy_toks_ws)

        # extract features and align them
        features = self.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
        features = features.squeeze(0)
        aligned_feats = alignment_utils.align_features_to_words(self, features, alignment)

        # wrap in spaCy Doc
        doc = Doc(
            nlp.vocab,
            words=['<s>'] + [x.text for x in spacy_toks] + ['</s>'],
            spaces=[True] + [x.endswith(' ') for x in spacy_toks_ws[:-1]] + [True, False],
        )
        assert len(doc) == aligned_feats.size(0)
        doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
        return doc

    def predict_hallucination_labels(self, inputs1: List[str], inputs2: List[str], first_seg_lengths: List[int]=None,
                                     raw=True, inputs_ref:List[str]=None):
        if raw:
            tokens = []
            first_seg_lengths = []
            second_seg_bpes = []
            if inputs_ref is not None:
                for i1, i2, i3 in zip(inputs1, inputs_ref, inputs2):
                    tt, tl, second_seg_bpe = self.encode(i1, i2, i3, raw=True)
                    tokens.append(tt)
                    first_seg_lengths.append(tl)
                    second_seg_bpes.append(second_seg_bpe)
            else:
                for i1, i2 in zip(inputs1, inputs2):
                    tt, tl, second_seg_bpe = self.encode(i1, i2, raw=True)
                    tokens.append(tt)
                    first_seg_lengths.append(tl)
                    second_seg_bpes.append(second_seg_bpe)
        else:
            tokens = [self.encode(i1, i2, raw=raw) for i1, i2 in zip(inputs1, inputs2)]
        sample = self._build_sample(tokens)

        full_lengths = torch.LongTensor([x.numel() for x in tokens]).to(self.device)
        first_seg_lengths = torch.LongTensor(first_seg_lengths).to(self.device)
        # target are stripped, offset labels for target sentences only
        # directly concat as the target in above
        mask = torch.arange(sample.size(1)).to(self.device).unsqueeze(0).expand(sample.size(0), sample.size(1))
        # first_seg_lengths+1 is the start of the target labels;
        # full_lengths-1 is the last token, -2 is the actual last token of target w/o <eos>
        target_mask = (mask.ge(first_seg_lengths.unsqueeze(1))) & (mask.le(full_lengths.unsqueeze(1) - 2))
        if sample.size(-1) > self.model.max_positions():
            print("hi")
            sample = sample[:, :self.model.max_positions()]
            target_mask = target_mask[:, :self.model.max_positions()]

        logits, _ = self.model(sample, features_only=True, classification_head_name='sentence_classification_head',
            target_mask=target_mask)
        lprobs = F.log_softmax(logits, dim=-1)
        predictions = lprobs.argmax(-1).cpu().numpy()
        hallucination_probs = torch.exp(lprobs)[:, 1].cpu().numpy()
        if raw:
            return predictions, hallucination_probs, second_seg_bpes
        else:
            return predictions, hallucination_probs

    def fill_noised_mask(self, masked_inputs: List[str], topk=1):
        masked_token = '<mask>'
        noises, topk_opt = [], []

        text_spans = [sent.split(masked_token) for src, sent in masked_inputs]
        noised_tokens = []
        targets_bpe = []
        for (src, _), segs in zip(masked_inputs, text_spans):
            bpe_src = self.bpe.encode(src.strip())
            bpe_tgt = ' {0} '.format(masked_token).join([self.bpe.encode(seg.rstrip()) for seg in segs])
            bpe_idx = self.task.source_dictionary.encode_line(
                '<s> ' + bpe_src + ' </s> </s> ' + bpe_tgt + ' </s>',
                append_eos=False,
                add_if_not_exist=False,
            )
            tgt_bpe_idx = self.task.source_dictionary.encode_line(
                '<s> ' + bpe_tgt + ' </s>',
                append_eos=False,
                add_if_not_exist=False,
            )
            noised_tokens.append(bpe_idx)
            targets_bpe.append(tgt_bpe_idx)

        sample = self._build_sample(noised_tokens).long()
        masked_index = (sample == self.task.mask_idx)

        with utils.eval(self.model):
            # features: B x T x |V|
            features, extra = self.model(
                sample,
                features_only=False,
                return_all_hiddens=False,
                masked_tokens=masked_index
            )
        prob = features.softmax(dim=-1)
        # values, index = prob.topk(k=topk, dim=-1)
        values, index = prob.max(dim=-1)
        index = index.squeeze(-1)  # K
        extra_symbols_to_ignore = set([])
        extra_symbols_to_ignore.add(self.task.source_dictionary[self.task.source_dictionary.eos()])
        extra_symbols_to_ignore.add(self.task.source_dictionary[self.task.source_dictionary.bos()])

        tot_masks = 0
        for ii, sent in enumerate(targets_bpe):
            decode_noise_tokens = self.decode(sent)
            decode_noise_tokens = decode_noise_tokens.replace("<mask>", " <mask>").strip()
            K = masked_index[ii, :].sum().item()
            topk_predictions = index[tot_masks : tot_masks+K]
            tot_masks += K
            assert len(topk_predictions) == decode_noise_tokens.split(" ").count('<mask>')
            output = []
            mask_count = 0
            topk_predicted_token_bpe = self.task.source_dictionary.string(topk_predictions, skip_ignore=True).split()
            for token in decode_noise_tokens.split(" "):
                if token == "<mask>":
                    predict_bpe = topk_predicted_token_bpe[mask_count]
                    if predict_bpe in extra_symbols_to_ignore:
                        continue
                    predicted_token = self.bpe.decode(predict_bpe)
                    # output.append("[" + predicted_token.strip() + "]")
                    output.append(predicted_token.strip())
                    mask_count += 1
                else:
                    output.append(token.strip())
            topk_opt.append(" ".join(output))
            noises.append(decode_noise_tokens)
        return topk_opt, noises

    def fill_mask(self, masked_input: str, topk: int = 5):
        masked_token = '<mask>'
        assert masked_token in masked_input and masked_input.count(masked_token) == 1, \
            "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)

        text_spans = masked_input.split(masked_token)
        text_spans_bpe = (' {0} '.format(masked_token)).join(
            [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
        ).strip()
        tokens = self.task.source_dictionary.encode_line(
            '<s> ' + text_spans_bpe + ' </s>',
            append_eos=False,
            add_if_not_exist=False,
        )

        masked_index = (tokens == self.task.mask_idx).nonzero()
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with utils.eval(self.model):
            features, extra = self.model(
                tokens.long().to(device=self.device),
                features_only=False,
                return_all_hiddens=False,
            )
        logits = features[0, masked_index, :].squeeze()
        prob = logits.softmax(dim=0)
        values, index = prob.topk(k=topk, dim=0)
        topk_predicted_token_bpe = self.task.source_dictionary.string(index)

        topk_filled_outputs = []
        for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(' ')):
            predicted_token = self.bpe.decode(predicted_token_bpe)
            # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
            if predicted_token_bpe.startswith('\u2581'):
                predicted_token = ' ' + predicted_token
            if " {0}".format(masked_token) in masked_input:
                topk_filled_outputs.append((
                    masked_input.replace(
                        ' {0}'.format(masked_token), predicted_token
                    ),
                    values[index].item(),
                    predicted_token,
                ))
            else:
                topk_filled_outputs.append((
                    masked_input.replace(masked_token, predicted_token),
                    values[index].item(),
                    predicted_token,
                ))
        return topk_filled_outputs

    def disambiguate_pronoun(self, sentence: str) -> bool:
        """
        Usage::

            >>> disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
            True

            >>> disambiguate_pronoun('The trophy would not fit in the brown suitcase because [it] was too big.')
            'The trophy'
        """
        assert hasattr(self.task, 'disambiguate_pronoun'), \
            'roberta.disambiguate_pronoun() requires a model trained with the WSC task.'
        with utils.eval(self.model):
            return self.task.disambiguate_pronoun(self.model, sentence, use_cuda=self.device.type == 'cuda')

    def load_noise_hyperparam(self, noise_params):
        self.noise = True
        self.noise_params = noise_params
        self.mask_idx = self.task.mask_idx
        self.vocab_length = len(self.task.source_dictionary)
        if self.noise_params['mask_whole_word']:
            self.noise_params['mask_whole_word'] = get_whole_word_mask(self.args, self.task.ource_dictionary)
        else:
            self.noise_params['mask_whole_word'] = None