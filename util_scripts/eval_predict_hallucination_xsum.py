from fairseq.models.roberta import RobertaModel
import os
import sys
import numpy as np
import shutil
from scipy import stats
import torch

test_dirs = ["eval_data/xsum/PtGen", "eval_data/xsum/TConvS2S", "eval_data/xsum/TranS2S", "eval_data/xsum/BERTS2S"]
test_prefix = ["PtGen", "TConvS2S", "TranS2S", "BERTS2S"]

models = ["path/to/the/saved/model"]  # you can test multiple models
datapath = "path/to/train/data"
opt_dir = "output_logs"
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)


def make_batches(total, bsz):
    batches = []
    for ii in range(0, total, bsz):
        batches.append((ii, ii+bsz if ii + bsz < total else total))
    return batches


def convert_gpt2_labels_to_raw_labels(sent_bpe, sent_detoks, bpe_labels):
    cat_bpe = sent_bpe.split()
    assert len(cat_bpe) == len(bpe_labels)
    detok_labels = []
    detoks = []

    atom = []
    labels = []
    for token, label in zip(cat_bpe, bpe_labels):
        debped_token = roberta.bpe.decode(token)
        # print(token, debped_token)
        # input()
        if len(atom) == 0:
            atom.append(debped_token)
            labels.append(label)
        elif debped_token.startswith(' ') and len(atom) > 0:
            detok_labels.append(1 if sum(labels) > 0 else 0)
            recover = "".join(atom).strip()
            detoks.append(recover)

            atom = []
            atom.append(debped_token)
            labels = []
            labels.append(label)
        else:
            atom.append(debped_token)
            labels.append(label)

    if len(atom) > 0 and len(labels) > 0:
        detok_labels.append(1 if sum(labels) > 0 else 0)
        token = "".join(atom).strip()
        detoks.append(token)
    assert len(sent_detoks) == len(detok_labels)
    # assert " ".join(detoks) == " ".join(sent_detoks)
    return detok_labels

# batch size
bsz = 100

for model in models:
    print(model)

    roberta = RobertaModel.from_pretrained(
        model,
        checkpoint_file='checkpoint.pt',
        data_name_or_path=datapath
    )
    raw = True
    print("Loaded the model!")
    possible_correct_all, ncorrect, nsamples = 0, 0, 0
    roberta.cuda()
    roberta.eval()
    roberta.half()
    max_positions = roberta.model.max_positions()

    for use_ref in [0, 1]:
        print(f"use ref = {use_ref}")

        for prefix, test_dir in zip(test_prefix, test_dirs):
            log_name = os.path.join(opt_dir, "use_ref_{}_{}.log".format(use_ref, prefix.lower()))
            flog = open(log_name, "w", encoding="utf-8")
            print(prefix)

            count = 0

            gold_correct = 0
            nh_correct = 0
            recall_total = 0
            precision_total = 0
            tot_tokens = 0

            sent_pred_labels = []
            all_sent_labels_gold = []

            predict_hallucination_strengths_by_token = []
            predict_hallucination_strengths_by_probs = []
            gold_hallucination_strengths = []

            data = []
            with open(os.path.join(test_dir, prefix + ".source"), encoding='utf-8') as fsrc, \
                    open(os.path.join(test_dir, prefix + ".target"), encoding='utf-8') as ftgt, \
                    open(os.path.join(test_dir, prefix + ".label"), encoding='utf-8') as flabel, \
                    open(os.path.join(test_dir, prefix + ".ref"), encoding='utf-8') as fin_ref:

                for src, tgt, label, ref in zip(fsrc, ftgt, flabel, fin_ref):
                    data.append((src.strip(), tgt.strip(), label.strip(), ref.strip()))

            for i, j in make_batches(len(data), bsz):
                slines = [[sample[0] for sample in data[i: j]], [sample[1] for sample in data[i: j]],
                          [sample[2] for sample in data[i: j]], [sample[3] for sample in data[i: j]]]
                count += len(slines[0])

                sent_target_labels = np.array(
                    [1 if sum([int(l) for l in s.strip().split()]) > 0 else 0 for s in slines[2]])
                all_sent_labels_gold.extend(sent_target_labels)
                target = np.array([float(l) for s in slines[2] for l in s.strip().split()])

                gold_strengths = [sum([int(l) for l in s.strip().split()]) * 1.0 / len(s.strip().split()) for s in
                                  slines[2]]
                gold_hallucination_strengths.extend(gold_strengths)
                gold_correct += sum([sum([int(l) for l in s.strip().split()]) * 1.0 for s in slines[2]])

                with torch.no_grad():
                    prediction_label, prediction_probs, target_bpes = roberta.predict_hallucination_labels(slines[0],
                                                                                                           slines[1],
                                                                                                           raw=raw,
                                                                                                           inputs_ref=slines[-1] if use_ref else None)
                # convert bpe labels to raw labels
                new_prediction_labels = []
                new_target = []
                full_bpes = [bpe for sent in target_bpes for bpe in sent.split()]
                assert len(full_bpes) == len(prediction_label)
                cum_lengths = 0
                for idx, (raw_target, sent) in enumerate(zip(slines[1], target_bpes)):
                    token_prediction_labels = convert_gpt2_labels_to_raw_labels(sent,
                                                                                raw_target.split(),
                                                                                prediction_label[
                                                                                cum_lengths:cum_lengths + len(
                                                                                    sent.split())])
                    sent_pred_labels.append(int(sum(token_prediction_labels) > 0))
                    predict_hallucination_strengths_by_token.append(
                        float(sum(token_prediction_labels)) / len(token_prediction_labels))
                    predict_hallucination_strengths_by_probs.append(
                        np.mean(prediction_probs[cum_lengths:cum_lengths + len(sent.split())]))

                    cum_lengths += len(sent.split())
                    flog.write("Reference: " + slines[-1][idx] + '\n')
                    flog.write("Ground-Truth: " + " ".join(["{}[{}]".format(t, p) for t, p in zip(raw_target.split(),
                                                                                                  slines[2][
                                                                                                      idx].strip().split())]) + '\n')
                    flog.write("Prediction: " + " ".join(
                        ["{}[{}]".format(t, p) for t, p in zip(raw_target.split(), token_prediction_labels)]) + "\n\n")
                    new_prediction_labels.extend(token_prediction_labels)
                prediction_label = np.array(new_prediction_labels)

                nh_correct += sum([1 for p, t in zip(prediction_label, target) if p == 1 and t == 1])
                recall_total += sum(target == 1)
                precision_total += sum(prediction_label == 1)
                tot_tokens += len(prediction_label)
                ncorrect += sum(prediction_label == target)
                nsamples += len(target)

                if count > 0 and count % 100 == 0:
                    print("Processed {} lines!".format(count))

            acc = float(ncorrect)/float(nsamples)
            recall = float(nh_correct)/float(recall_total)
            precision = float(nh_correct)/float(precision_total)
            f1 = 2 * precision * recall / (precision + recall)

            sent_pred_labels = np.array(sent_pred_labels)
            sent_labels_gold = np.array(all_sent_labels_gold)
            assert len(all_sent_labels_gold) == len(sent_pred_labels)
            sent_acc = sum(sent_labels_gold == sent_pred_labels) * 1.0 / len(sent_labels_gold)

            index = (sent_labels_gold == 1)
            sent_correct = sum(sent_labels_gold[index] == sent_pred_labels[index])
            sent_precision_denom = sum(sent_pred_labels == 1)
            sent_precision = sent_correct * 1. / sent_precision_denom
            sent_recall = sent_correct * 1. / sum(index)
            sent_f1 = 2 * sent_precision * sent_recall / (sent_precision + sent_recall)

            spearman_corr_by_token, p_value_by_token = stats.spearmanr(gold_hallucination_strengths, predict_hallucination_strengths_by_token)
            spearman_corr_by_probs, p_value_by_probs = stats.spearmanr(gold_hallucination_strengths, predict_hallucination_strengths_by_probs)

            flog.write('Sentence-level F1: {}\n'.format(sent_f1))
            flog.write('Sentence-level accuracy: {}'.format(sent_acc) + '\n')
            flog.write('Spearman-corr by token: {}\n'.format(spearman_corr_by_token))
            flog.write('Spearman-corr by probs: {}\n'.format(spearman_corr_by_probs))
            flog.write("Percentage of hallucination tokens = {}, gold = {}\n".format(precision_total * 1.0 / tot_tokens, gold_correct/tot_tokens))
            flog.write('| {} | Accuracy: {}, recall: {}, precision: {}, f1: {}'.format(prefix, acc, recall, precision, f1) + '\n')

            print("Percentage of hallucination tokens = {}, gold = {}".format(precision_total*1.0/tot_tokens, gold_correct/tot_tokens))
            print('Sentence-level F1: {}'.format(sent_f1))
            print("Sentence-level hallucination percentage (gold) = {}".format(sum(sent_labels_gold)*1.0/len(sent_labels_gold)))
            print('Spearman-corr by token: {}'.format(spearman_corr_by_token))
            print('Spearman-corr by probs: {}'.format(spearman_corr_by_probs))
            print(precision, recall, f1, sent_acc, spearman_corr_by_token, spearman_corr_by_probs, acc)

            flog.close()
