from fairseq.models.roberta import XLMRModel
import os
import sys
import numpy as np
import shutil
from scipy import stats
import torch
import re
from sacremoses import MosesTokenizer, MosesDetokenizer
md = MosesDetokenizer(lang='en')

raw_dir = "input_dir"
source_fname = "source"  # file name of the source input
hypo_fname = "hypo"  # file name that you want to predict if they contain hallucinations based on the source

model_path = "path/to/the/saved/model"
datapath = "path/to/train/data"
opt_dir = "outputs"
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)
print("opt dir: " + opt_dir)
flog = open(os.path.join(opt_dir, "hal_pred.log"), "w", encoding="utf-8")
log_path = os.path.join(opt_dir, "gen.log")

# read input in
data = []
with open(os.path.join(raw_dir, source_fname), encoding='utf-8') as fin1, \
        open(os.path.join(raw_dir, hypo_fname), encoding='utf-8') as fin2:
    for l1, l2 in zip(fin1, fin2):
        data.append((l1.strip(), l2.strip()))


def make_batches(total, bsz):
    batches = []
    for ii in range(0, total, bsz):
        batches.append((ii, ii+bsz if ii + bsz < total else total))
    return batches


print(model_path)
xlmr = XLMRModel.from_pretrained(
    model_path,
    checkpoint_file='checkpoint.pt',
    data_name_or_path=datapath
)

print("Loaded the model!")
xlmr.cuda()
xlmr.eval()

max_positions = xlmr.model.max_positions()
use_ref = 0
print(f"use ref = {use_ref}")


def convert_spm_labels_to_raw_labels(sent_bpe, sent_detoks, bpe_labels):
    cat_bpe = sent_bpe.split()
    assert len(cat_bpe) == len(bpe_labels)
    detok_labels = []
    detoks = []
    atom = []
    labels = []
    for token, label in zip(cat_bpe, bpe_labels):
        if len(atom) == 0:
            atom.append(token)
            labels.append(label)
        elif token.startswith('\u2581') and len(atom) > 0:
            detok_labels.append(1 if sum(labels) > 0 else 0)
            recover = " ".join(atom).replace('\u2581', ' ').replace(' ', '')
            detoks.append(recover)
            atom = []
            atom.append(token)
            labels = []
            labels.append(label)
        else:
            atom.append(token)
            labels.append(label)
    if len(atom) > 0 and len(labels) > 0:
        detok_labels.append(1 if sum(labels) > 0 else 0)
        token = " ".join(atom).replace('\u2581', ' ').replace(' ', '')
        detoks.append(token)

    # we don't deal with overlong sentences
    if len(sent_detoks) != len(detok_labels):
        return [1] * len(sent_detoks), True
    return detok_labels, False


bsz = 300
count = 0
tot_tokens = 0
tot_pred_hal_tokens = 0

sent_pred_labels = []
prediction_token_labels = []
bad_num = 0


for i, j in make_batches(len(data), bsz):
    slines = [[sample[0] for sample in data[i: j]], [sample[1] for sample in data[i: j]]]
    first_seg_lengths = None
    # if raw, target are detoknized labels
    with torch.no_grad():
        src, tgt = slines[0], slines[1]
        # maybe open these hacks
        # src = [s.lower() for s in slines[0]]
        # tgt = [t.lower() for t in slines[1]]
        prediction_label, prediction_probs, target_bpes = xlmr.predict_hallucination_labels(src, tgt,
                                                                          first_seg_lengths=first_seg_lengths,
                                                                          raw=True,
                                                                          inputs_ref=None)
    # convert bpe labels to annotation raw labels
    full_bpes = [bpe for sent in target_bpes for bpe in sent.split()]
    assert len(full_bpes) == len(prediction_label)

    cum_lengths = 0
    for idx, (raw_source, raw_target, sent) in enumerate(zip(slines[0], slines[1], target_bpes)):
        # hack for Chinese special characters
        raw_target = raw_target.replace("℃", "°C")
        token_prediction_labels, is_bad = convert_spm_labels_to_raw_labels(sent,
                                                                raw_target.split(),
                                                                prediction_label[cum_lengths:cum_lengths+len(sent.split())])
        if raw_source.lower() == raw_target.lower():
            token_prediction_labels = [0] * len(raw_target.split())

        sent_pred_labels.append(int(sum(token_prediction_labels) > 0))
        cum_lengths += len(sent.split())
        flog.write("Source: " + slines[0][idx] + '\n')
        flog.write("Token-Prediction: " + " ".join(["{}[{}]".format(t, p) for t, p in zip(raw_target.split(), token_prediction_labels)]) + "\n\n")
        prediction_token_labels.append(token_prediction_labels)

        if is_bad:
            bad_num += 1
        else:
            tot_pred_hal_tokens += sum(np.array(token_prediction_labels) == 1)
            tot_tokens += len(token_prediction_labels)

    count += 1
    if count > 0 and count % 100 == 0:
        print("Processed {} batches, bad {}!".format(count, bad_num))

flog.close()

print("Bad sentences = {}, Percentage of hallucinations = {}".format(bad_num, tot_pred_hal_tokens*1.0/tot_tokens))
