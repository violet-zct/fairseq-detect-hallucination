from fairseq.models.roberta import XLMRModel
import os
import sys
import numpy as np
import shutil
from scipy import stats
import torch

test_dirs = ["/home/chuntinz/tir5/data/qe_wmt18_ende/data2"]
test_prefix = ["test"]

model_root = "/home/chuntinz/tir5/fairseq-detect-hallucination/checkpoints/"
models = [os.path.join(model_root, sys.argv[1])]
datapath = "/home/chuntinz/tir5/data/qe_wmt18_ende/data2/bart_gen/sup/bin"
opt_dir = "output_logs"
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)


def make_batches(total, bsz):
    batches = []
    for ii in range(0, total, bsz):
        batches.append((ii, ii+bsz if ii + bsz < total else total))
    return batches


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
    assert len(sent_detoks) == len(detok_labels)
    # assert " ".join(detoks) == " ".join(sent_detoks)
    return detok_labels

# batch size
bsz = 100

for model in models:
    print(model)

    xlmr = XLMRModel.from_pretrained(
        model,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=datapath
    )

    raw = True
    print("Loaded the model!")
    xlmr.cuda()
    xlmr.eval()
    max_positions = xlmr.model.max_positions()

    for use_ref in [0, 1]:
        print(f"use ref = {use_ref}")

        for prefix, test_dir in zip(test_prefix, test_dirs):
            print(prefix, test_dir)
            log_name = os.path.join(opt_dir, "{}_use_ref_{}.log".format(model.split("/")[-1], use_ref))
            flog = open(log_name, "w", encoding="utf-8")

            ncorrect, nsamples = 0, 0
            count = 0

            gold_correct = 0
            nh_correct = 0
            recall_total = 0
            precision_total = 0

            nf_correct = nf_recall_total = nf_precision_total = 0
            tot_tokens = 0

            sent_pred_labels = []
            all_sent_labels_gold = []

            predict_hallucination_strengths_by_token = []
            predict_hallucination_strengths_by_probs = []
            gold_hallucination_strengths = []

            data = []
            with open(os.path.join(test_dir, prefix+".en"), encoding='utf-8') as fsrc, \
                    open(os.path.join(test_dir, prefix+".de"), encoding='utf-8') as ftgt, \
                    open(os.path.join(test_dir, prefix+".tok.labels"), encoding='utf-8') as flabel, \
                    open(os.path.join(test_dir, prefix + ".ref"), encoding='utf-8') as fin_ref:

                for src, tgt, label, ref in zip(fsrc, ftgt, flabel, fin_ref):
                    data.append((src.strip(), tgt.strip(), label.strip(), ref.strip()))

            for i, j in make_batches(len(data), bsz):
                slines = [[sample[0] for sample in data[i: j]], [sample[1] for sample in data[i: j]],
                          [sample[2] for sample in data[i: j]], [sample[3] for sample in data[i: j]]]
                count += len(slines[0])

                sent_target_labels = np.array([1 if sum([int(l) for l in s.strip().split()]) > 0 else 0 for s in slines[2]])
                all_sent_labels_gold.extend(sent_target_labels)
                target = np.array([float(l) for s in slines[2] for l in s.strip().split()])

                gold_strengths = [sum([int(l) for l in s.strip().split()]) * 1.0 / len(s.strip().split()) for s in slines[2]]
                gold_hallucination_strengths.extend(gold_strengths)
                gold_correct += sum([sum([int(l) for l in s.strip().split()]) * 1.0 for s in slines[2]])

                with torch.no_grad():
                    prediction_label, prediction_probs, target_bpes = xlmr.predict_hallucination_labels(slines[0], slines[1],
                                                                                      raw=raw,
                                                                                      inputs_ref=slines[-1] if use_ref else None)
                # convert bpe labels to annotation raw labels
                new_prediction_labels = []
                new_target = []
                full_bpes = [bpe for sent in target_bpes for bpe in sent.split()]
                assert len(full_bpes) == len(prediction_label)
                cum_lengths = 0
                for idx, (raw_target, sent) in enumerate(zip(slines[1], target_bpes)):
                    token_prediction_labels = convert_spm_labels_to_raw_labels(sent,
                                                                            raw_target.split(),
                                                                            prediction_label[cum_lengths:cum_lengths+len(sent.split())])
                    sent_pred_labels.append(int(sum(token_prediction_labels) > 0))
                    predict_hallucination_strengths_by_token.append(float(sum(token_prediction_labels))/len(token_prediction_labels))
                    predict_hallucination_strengths_by_probs.append(np.mean(prediction_probs[cum_lengths:cum_lengths+len(sent.split())]))
                    cum_lengths += len(sent.split())
                    flog.write("Reference: " + slines[-1][idx] + '\n')
                    flog.write("Ground-Truth: " + " ".join(["{}[{}]".format(t, p) for t, p in zip(raw_target.split(), slines[2][idx].strip().split())]) + '\n')
                    flog.write("Prediction: " + " ".join(["{}[{}]".format(t, p) for t, p in zip(raw_target.split(), token_prediction_labels)]) + "\n\n")
                    new_prediction_labels.extend(token_prediction_labels)
                prediction_label = np.array(new_prediction_labels)

                assert len(prediction_label) == len(target)
                nh_correct += sum([1 for p, t in zip(prediction_label, target) if p == 1 and t == 1])
                recall_total += sum(target == 1)
                precision_total += sum(prediction_label == 1)

                nf_correct += sum([1 for p, t in zip(prediction_label, target) if p == 0 and t == 0])
                nf_recall_total += sum(target == 0)
                nf_precision_total += sum(prediction_label == 0)

                tot_tokens += len(prediction_label)
                ncorrect += sum(prediction_label == target)
                nsamples += len(target)

                if count > 0 and count % 100 == 0:
                    print("Processed {} lines!".format(count))

            acc = float(ncorrect)/float(nsamples)
            recall = float(nh_correct)/float(recall_total)
            precision = float(nh_correct)/float(precision_total)
            f1 = 2 * precision * recall / (precision + recall)

            nf_recall = float(nf_correct) / float(nf_recall_total)
            nf_precision = float(nf_correct) / float(nf_precision_total)
            nf_f1 = 2 * nf_precision * nf_recall / (nf_precision + nf_recall)

            f1_mult = f1 * nf_f1

            flog.write("Percentage of hallucination tokens = {}, gold percentage = {}\n".format(precision_total * 1.0 / tot_tokens, gold_correct / tot_tokens))
            flog.write('| {} | Accuracy: {}, recall: {}, precision: {}'.format(prefix, acc, recall, precision) + '\n')
            flog.write('| nt_f1 = {}, nf_f1 = {}, f1_mult = {}\n'.format(f1, nf_f1, f1_mult))
            flog.close()

            print("Percentage of hallucination tokens = {}, gold = {}".format(precision_total * 1.0 / tot_tokens, gold_correct / tot_tokens))
            print(('| acc = {}, nt_f1 = {}, nf_f1 = {}, f1_mult = {}\n'.format(acc, f1, nf_f1, f1_mult)))
