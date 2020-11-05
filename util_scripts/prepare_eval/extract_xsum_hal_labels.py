import os
import sys
import csv
from collections import defaultdict
import numpy as np

# local run
raw_dir = "/Users/chuntinz/Desktop/xsum"
google_annot = "/Users/chuntinz/Documents/research/fairseq-hallucination/local/hallucination_annotations_xsum_summaries.csv"
fact_annot = "/Users/chuntinz/Documents/research/fairseq-hallucination/local/factuality_annotations_xsum_summaries.csv"


def checkInput(rate, n):
    """
    Check correctness of the input matrix
    @param rate - ratings matrix
    @return n - number of raters
    @throws AssertionError
    """
    N = len(rate)
    k = len(rate[0])
    assert all(len(rate[i]) == k for i in range(k)), "Row length != #categories)"
    # assert all(isinstance(rate[i][j], int) for i in range(N) for j in range(k)), "Element not integer"
    assert all(sum(row) == n for row in rate), "Sum of ratings != #raters)"


def fleissKappa(rate, n):
    """
    Computes the Kappa value
    @param rate - ratings matrix containing number of ratings for each subject per category
    [size - N X k where N = #subjects and k = #categories]
    @param n - number of raters
    @return fleiss' kappa
    """

    N = len(rate)
    k = len(rate[0])
    print("#raters = ", n, ", #subjects = ", N, ", #categories = ", k)
    checkInput(rate, n)

    # mean of the extent to which raters agree for the ith subject
    PA = sum([(sum([i ** 2 for i in row]) - n) / (n * (n - 1)) for row in rate]) / N
    print("PA = ", PA)

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j ** 2 for j in [sum([rows[i] for rows in rate]) / (N * n) for i in range(k)]])
    print("PE =", PE)

    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)

    return kappa


def cal_token_level_kappa(labels_1, labels_2, labels_3):
    # print(len(labels_1), len(labels_2), len(labels_3))
    all_tokens = sum([len(ll) for ll in labels_1])
    # print(all_tokens, sum([len(ll) for ll in labels_2]), sum([len(ll) for ll in labels_3]))
    k = 2
    n = 3
    mat = np.zeros((all_tokens, k))
    idx = 0
    for l1, l2, l3 in zip(labels_1, labels_2, labels_3):
        for t1, t2, t3 in zip(l1, l2, l3):
            mat[idx, int(t1)] += 1
            mat[idx, int(t2)] += 1
            mat[idx, int(t3)] += 1
            idx += 1

    print(mat.shape)
    return fleissKappa(mat, n)


def read_data():
    data = {}
    with open(os.path.join(raw_dir, 'test.summary'), encoding='utf-8') as fs, \
        open(os.path.join(raw_dir, 'test.document'), encoding='utf-8') as fd, \
        open(os.path.join(raw_dir, 'test.docid'), encoding='utf-8') as fi:
        for doc, summary, docid in zip(fd, fs, fi):
            data[docid.strip()] = (doc.strip(), summary.strip())
    return data


def read_hal_csv():
    eval_data = {}
    line_no = 0
    with open(google_annot, "r", encoding='utf-8') as fin:
        csv_reader = csv.reader(fin, delimiter=',')

        for fields in csv_reader:
            assert len(fields) == 6
            if line_no == 0:
                line_no += 1
                continue
            doc_id = fields[0]
            model = fields[1]
            summary = fields[2]
            hallucination_type = fields[-3]
            hallucination_span = fields[-2]
            work_id = int(fields[-1].split("_")[-1])

            if model not in eval_data:
                eval_data[model] = defaultdict(list)
            eval_data[model][doc_id].append((summary, work_id, hallucination_span, hallucination_type))
    return eval_data


def read_fact_csv():
    eval_data = {}
    line_no = 0
    with open(fact_annot, "r", encoding='utf-8') as fin:
        csv_reader = csv.reader(fin, delimiter=',')

        for fields in csv_reader:
            assert len(fields) == 5
            if line_no == 0:
                line_no += 1
                continue
            doc_id = fields[0]
            model = fields[1]
            summary = fields[2]
            is_factual = fields[3]
            work_id = int(fields[-1].split("_")[-1])

            if model not in eval_data:
                eval_data[model] = defaultdict(list)
            eval_data[model][doc_id].append((summary, work_id, is_factual))
    return eval_data


def convert_factuality_to_dict(factuality_annot):
    worker_annot = dict()
    for summary, work_id, is_factual in factuality_annot:
        if work_id in worker_annot:
            print("Error, same evaluator has duplicate rating of factuality of the same summary!")
        print(is_factual)
        worker_annot[work_id] = 1 if is_factual == "yes" else 0
    return worker_annot


def aggregate(annotations, factuality_annot=None):
    summary = annotations[0][0].strip().split()
    spans = [(work_id, span) for summary, work_id, span, type in annotations]
    word_counters = np.zeros(len(summary))

    if factuality_annot is not None:
        separate_factuality_labels = convert_factuality_to_dict(factuality_annot)
    else:
        separate_factuality_labels = None

    separate_labels = {ii: np.zeros(len(summary)) for ii in range(3)}
    for worker_id, span in spans:
        if span == 'NULL':
            continue

        pointer = 0
        span_length = len(span.split())
        while True:
            if span in " ".join(summary[pointer: pointer+span_length]):
                label = 1
                for idx in range(pointer, pointer+span_length):
                    word_counters[idx] += label
                    separate_labels[worker_id][idx] = label
                pointer += span_length
            else:
                pointer += 1
            if pointer + span_length > len(summary):
                break

    nonzero = len([1 for wc in word_counters if wc > 0])
    labels = [1 if count >= 2 else 0 for idx, count in enumerate(word_counters)]
    agreed_labels = sum(labels)

    return labels, agreed_labels, nonzero, separate_labels, " ".join(summary)


def cal_f1(golds, preds):
    correct_words = 0
    recall_total = 0
    precision_total = 0
    for gold, pred in zip(golds, preds):
        assert len(gold) == len(pred)
        correct_words += sum([1 if g == 1 and p == 1 else 0 for g, p in zip(gold, pred)])
        recall_total += sum(np.array(gold) == 1)
        precision_total += sum(np.array(pred) == 1)
    precision = correct_words * 1.0 / precision_total
    recall = correct_words * 1.0 / recall_total
    f1 = 2 * precision * recall / (precision + recall)
    print(precision, recall, f1)
    return precision, recall, f1


test_data = read_data()
hal_eval_data = read_hal_csv()
factual_eval_data = read_fact_csv()

for model in hal_eval_data.keys():
    output_dir = os.path.join("/Users/chuntinz/Documents/research/fairseq-hallucination/evals/xsum", model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    agreements = []
    hallucinations = 0
    agree_num, agree_denum = 0, 0
    all_labels = [[], [], []]
    aggregate_labels = []

    with open(os.path.join(output_dir, "{}.source".format(model)), "w", encoding='utf-8') as fs, \
            open(os.path.join(output_dir, "{}.target".format(model)), "w", encoding='utf-8') as ft, \
            open(os.path.join(output_dir, "{}.label".format(model)), "w", encoding='utf-8') as fl, \
            open(os.path.join(output_dir, "{}.ref".format(model)), "w", encoding='utf-8') as fr, \
            open(os.path.join(output_dir, "{}.docid".format(model)), "w", encoding='utf-8') as fdid:

        print(os.path.join(output_dir, "{}.label".format(model)))
        for docid in hal_eval_data[model].keys():
            # print(model, docid)
            labels, agreed_labels, total_hal_labels, separate_labels, generation = aggregate(hal_eval_data[model][docid],
                                                                                             factual_eval_data[model][docid] if model in factual_eval_data else None)

            for idx, sep_labels in separate_labels.items():
                all_labels[idx].append(sep_labels)

            doc, summary = test_data[docid]
            fs.write(doc + '\n')
            ft.write(generation + '\n')
            fl.write(" ".join([str(int(ll)) for ll in labels]) + '\n')

            fr.write(summary + '\n')
            fdid.write(docid + '\n')
            agree_num += agreed_labels
            agree_denum += total_hal_labels
            hallucinations = hallucinations + (1 if sum(labels) > 0 else 0)
            aggregate_labels.append(labels)

    cal_token_level_kappa(*all_labels)
    print(model, agree_num*1.0/agree_denum, hallucinations * 1.0 / len(all_labels[0]))

    p1, r1, f1 = cal_f1(aggregate_labels, all_labels[0])
    p2, r2, f2 = cal_f1(aggregate_labels, all_labels[1])
    p3, r3, f3 = cal_f1(aggregate_labels, all_labels[2])
    print("avg F1 = {}".format(np.mean([f1, f2, f3])))