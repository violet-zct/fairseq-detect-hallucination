from collections import defaultdict
import math
import numpy as np
import os
from nltk.corpus import wordnet as wn

root = "/private/home/chuntinz/work/assets/stanford-postagger-full-2020-08-06/hal_pos_ana"
root = "hal_pos_ana"


def read_pos_tags_and_tokenizations(path):
    tok_data = []
    tok_pos_tags = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            fields = line.strip().split()
            sent, tags = [], []
            for field in fields:
                if len(field.split("_")) != 2:
                    continue
                word, label = field.split("_")
                sent.append(word)
                tags.append(label)
            tok_data.append(sent)
            tok_pos_tags.append(tags)
    return tok_data, tok_pos_tags


def convert_raw_labels_to_token_labels(raw_sent, tok_sent, raw_labels):
    raw_words = raw_sent.strip().split()
    raw_labels = [int(rl) for rl in raw_labels.strip().split()]
    assert len(raw_words) == len(raw_labels)
    tok_labels = []

    temp_word = []
    temp_sent = []
    pointer = 0
    for tok in tok_sent:
        temp_word.append(tok)
        if "".join(temp_word) == raw_words[pointer]:
            tok_labels.extend([raw_labels[pointer]] * len(temp_word))
            temp_sent.append("".join(temp_word))
            pointer += 1
            temp_word = []

    assert " ".join(temp_sent) == " ".join(raw_words)
    assert len(tok_sent) == len(tok_labels)
    return tok_labels


def convert_token_labels_to_raw_labels(tok_labels, tok_sent, raw_sent):
    atom = []
    tags = []
    pointer = 0

    labels = []
    for tok, tag in zip(tok_sent, tok_labels):
        atom.append(tok)
        tags.append(tag)
        if "".join(atom) == raw_sent[pointer]:
            labels.append(tags[0])
            atom = []
            tags = []
            pointer += 1
    assert len(labels) == len(raw_sent)
    return labels


def mapping(tag):
    consider_tags_prefix = ["NN", "VB", "RB", "JJ"]
    maps = [wn.NOUN, wn.VERB, wn.ADV, wn.ADJ]
    # consider_tags_prefix = ["NN"]
    # maps = [wn.NOUN]

    for j in range(len(consider_tags_prefix)):
        if tag[:2] == consider_tags_prefix[j]:
            return maps[j]
    return None


def mapping_src(tag):
    consider_tags_prefix = ["NN", "VB", "RB", "JJ"]
    maps = ["n", "v", "r", "s"]
    # consider_tags_prefix = ["NN"]
    # maps = ["n"]
    for j in range(len(consider_tags_prefix)):
        if tag[:2] == consider_tags_prefix[j]:
            return maps[j]
    return None


def find_all_source_lemmas(tok_src_sent, tok_src_pos_tags):
    all_lemmas = set()
    for word, pos in zip(tok_src_sent, tok_src_pos_tags):
        tag = mapping_src(pos)
        if tag is not None:
            try:
                lemmas = wn.synset("{}.{}.01".format(word, tag)).lemma_names()
                all_lemmas.update(lemmas)
            except:
                pass
    return all_lemmas


def find_synonyms(tok_tgt_sent, tok_tgt_pos_tags, tok_src_sent, tok_src_pos_tags):
    all_src_lemma = find_all_source_lemmas(tok_src_sent, tok_src_pos_tags)
    is_synonyms = []
    all_synonyms = []
    for word, pos in zip(tok_tgt_sent, tok_tgt_pos_tags):
        tag = mapping(pos)
        if tag is None:
            is_synonyms.append(0)
            continue
        synonyms = wn.synsets(word, pos=tag)
        if len(synonyms) > 0:
            syn_lemmas = set([lemma for ss in synonyms for lemma in ss.lemma_names()])
            xx = syn_lemmas.intersection(all_src_lemma)
            all_synonyms.append(xx)
            if len(xx) > 0:
                is_synonyms.append(1)
            else:
                is_synonyms.append(0)
        else:
            is_synonyms.append(0)
    assert len(is_synonyms) == len(tok_tgt_sent)
    return is_synonyms, all_synonyms


prefix_list = ["BERTS2S.", "PtGen.", "TConvS2S.", "TranS2S."]

total_synonym = 0
pred_errors = 0
overlap_errors = 0

total_words = 0

for prefix in prefix_list:
    print(prefix)
    gold_raw_labels = open(os.path.join(root, prefix + "label")).readlines()
    pred_raw_labels = open(os.path.join(root, prefix + "pred.label")).readlines()
    # pred_raw_labels = open(os.path.join(root, prefix + "overlap.combine.pred.label"))
    overlap_xsum_labels = open(os.path.join(root, prefix + "overlap.label")).readlines()

    prediction_labels = []
    gold_labels = []

    raw_sents = open(os.path.join(root, prefix + "target"), "r", encoding="utf-8").readlines()
    tgt_tok_sents, tgt_tok_pos_tags = read_pos_tags_and_tokenizations(os.path.join(root, prefix.lower()+"target.pos"))
    src_tok_sents, src_tok_pos_tags = read_pos_tags_and_tokenizations(os.path.join(root, prefix.lower()+"source.pos"))

    fout = open(os.path.join("syn_preds", "{}syn.pred".format(prefix)), "w")
    for idx, (raw_sent, gold_raw_label, pred_raw_label, tgt_tok_sent, tgt_pos_tags, src_tok_sent, src_pos_tags) in \
            enumerate(zip(raw_sents, gold_raw_labels, pred_raw_labels, tgt_tok_sents, tgt_tok_pos_tags, src_tok_sents, src_tok_pos_tags)):

        gold_tok_label = convert_raw_labels_to_token_labels(raw_sent, tgt_tok_sent, gold_raw_label)
        pred_tok_label = convert_raw_labels_to_token_labels(raw_sent, tgt_tok_sent, pred_raw_label)
        assert len(gold_tok_label) == len(pred_tok_label) and len(pred_tok_label) == len(tgt_pos_tags)

        overlap_xsum_label = overlap_xsum_labels[idx]
        overlap_tok_labels = convert_raw_labels_to_token_labels(raw_sent, tgt_tok_sent, overlap_xsum_label)
        gold_labels.extend([int(ll) for ll in gold_raw_label.strip().split()])

        has_synonyms, synonyms = find_synonyms(tgt_tok_sent, tgt_pos_tags, src_tok_sent, src_pos_tags)

        pred_label = [1-ss for ss in has_synonyms]
        pred_syn_labels = convert_token_labels_to_raw_labels(pred_label, tgt_tok_sent, raw_sent.strip().split())

        fout.write(" ".join([str(ss) for ss in pred_syn_labels]) + "\n")
        prediction_labels.extend(pred_syn_labels)

        if sum(has_synonyms) > 0:
            total_synonym += sum(has_synonyms)
            for ii, (snym, gold_label) in enumerate(zip(has_synonyms, gold_tok_label)):
                if snym == 1 and gold_label == 0:
                    if overlap_tok_labels[ii] == 1:
                        overlap_errors += 1
                        # print("overlap error!")
                        # print(tgt_tok_sent[ii])
                        # print(src_tok_sent)
                        # input()

                    if pred_tok_label[ii] == 1:
                        pred_errors += 1
                        # print(tgt_tok_sent[ii])
                        # print(synonyms)
                        # print("pred error!")
                        # print(tgt_tok_sent[ii])
                        # print(src_tok_sent)
                        # input()
        total_words += len(gold_tok_label)
        if idx % 100 == 0:
            print("processed {} lines".format(idx))
    fout.close()
    prediction_labels = np.array(prediction_labels)
    gold_labels = np.array(gold_labels)
    ncorrect = sum([1 for p, t in zip(prediction_labels, gold_labels) if p == 1 and t == 1])
    precision_total = sum(prediction_labels == 1)
    recall_total = sum(gold_labels == 1)
    recall = ncorrect * 1.0 / recall_total
    precision = ncorrect * 1.0 / precision_total
    f1 = 2 * precision * recall / (precision + recall)

    print('precision = {}, recall = {}. F1 = {}'.format(precision, recall, f1))
    print('{} {} {}'.format(precision, recall, f1))

print("total words = {}, total synonyms = {}, pred_errors = {}, overlap pred errors = {}".format(total_words, total_synonym, pred_errors, overlap_errors))