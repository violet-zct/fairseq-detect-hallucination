import sys
import os
from sacremoses import MosesTokenizer, MosesDetokenizer
import re
import numpy as np
from collections import defaultdict

if len(sys.argv) != 2:
    print("Usage: python xx.py opt_dir")

md = MosesDetokenizer(lang='en')

# opt_dir = sys.argv[1]
opt_dir = "./evals/public_round2_v1"

if not os.path.exists(opt_dir):
    os.makedirs(opt_dir)

prefix = 'v1.r2'
model_types = ['at', 'mbart']
fout_source_paths = []
fout_ref_paths = []
fout_target_paths = []
fout_label_paths = []
fout_possible_label_paths = []

for mtype in model_types:
    fout_source_paths.append(os.path.join(opt_dir, "{}.{}.source".format(mtype, prefix)))
    fout_ref_paths.append(os.path.join(opt_dir, "{}.{}.ref".format(mtype, prefix)))
    fout_target_paths.append(os.path.join(opt_dir, "{}.{}.target".format(mtype, prefix)))
    fout_label_paths.append(os.path.join(opt_dir, "{}.{}.label".format(mtype, prefix)))
    fout_possible_label_paths.append(os.path.join(opt_dir, "{}.{}.pos.label".format(mtype, prefix)))


def read_annotation_labels(fname):
    line = open(fname).read().strip()
    cells = set()
    for field in line.split(")("):
        fields = field.rstrip(")").lstrip("(").split(",")
        cells.add((int(fields[0]), int(fields[1])))
    return cells


def detokenize_labels(tokenized_sent, tokenized_labels):
    AGGRESSIVE_HYPHEN_SPLIT = r" \@\-\@ ", r"-"

    # Unescape special characters.
    UNESCAPE_FACTOR_SEPARATOR = r"&#124;", r"|"
    UNESCAPE_LEFT_ANGLE_BRACKET = r"&lt;", r"<"
    UNESCAPE_RIGHT_ANGLE_BRACKET = r"&gt;", r">"
    UNESCAPE_DOUBLE_QUOTE = r"&quot;", r'"'
    UNESCAPE_SINGLE_QUOTE = r"&apos;", r"'"
    UNESCAPE_SYNTAX_NONTERMINAL_LEFT = r"&#91;", r"["
    UNESCAPE_SYNTAX_NONTERMINAL_RIGHT = r"&#93;", r"]"
    UNESCAPE_AMPERSAND = r"&amp;", r"&"
    # The legacy regexes are used to support outputs from older Moses versions.
    UNESCAPE_FACTOR_SEPARATOR_LEGACY = r"&bar;", r"|"
    UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY = r"&bra;", r"["
    UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY = r"&ket;", r"]"

    MOSES_UNESCAPE_XML_REGEXES = [
        AGGRESSIVE_HYPHEN_SPLIT,
        UNESCAPE_FACTOR_SEPARATOR_LEGACY,
        UNESCAPE_FACTOR_SEPARATOR,
        UNESCAPE_LEFT_ANGLE_BRACKET,
        UNESCAPE_RIGHT_ANGLE_BRACKET,
        UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY,
        UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY,
        UNESCAPE_DOUBLE_QUOTE,
        UNESCAPE_SINGLE_QUOTE,
        UNESCAPE_SYNTAX_NONTERMINAL_LEFT,
        UNESCAPE_SYNTAX_NONTERMINAL_RIGHT,
        UNESCAPE_AMPERSAND,
    ]

    detok_sent = md.detokenize(tokenized_sent).split()
    # print(detok_sent)
    # input()
    detok_labels = []
    detok_pointer = 0
    labels = []
    match_quotes = 0
    atom = []
    for ii, token in enumerate(tokenized_sent):
        for regexp, substitution in MOSES_UNESCAPE_XML_REGEXES:
            token = re.sub(regexp, substitution, token)
        atom.append(token)
        labels.append(tokenized_labels[ii])
        if "".join(atom) == detok_sent[detok_pointer]:
            detok_pointer += 1
            labels.append(tokenized_labels[ii])
            detok_labels.append(1 if sum(labels) > 0 else 0)

            labels = []
            atom = []

    assert detok_pointer == len(detok_sent)
    return detok_sent, detok_labels


def read_sheet(flabel, fsheet, fmeta, detoked=False):
    activate_cells = read_annotation_labels(flabel)
    total = 0
    result_blocks = {mtype:[] for mtype in model_types}
    total_bad_annotate = 0
    meta_data = open(fmeta, "r")
    _ = meta_data.readline()
    with open(fsheet, "r", encoding="utf-8") as fin:
        row = -1
        while True:
            line = fin.readline()
            row += 1
            if line == '':
                break
            fields = line.strip().split('\t')
            if fields[0] == 'Source:':
                block = []
                # source
                block.append(fields[1].strip() + '\n')

                fields = fin.readline().strip().split('\t')
                row += 1
                assert fields[0] == 'Reference:'
                # target
                block.append(fields[1].strip() + '\n')

                # detokenized translation
                fields = fin.readline().strip().split('\t')
                tok_target = fields[1].strip().split()

                if len(fields) > 2:
                    bad_annot = True
                    total_bad_annotate += 1
                else:
                    bad_annot = False
                num_tokens = len(tok_target)
                row += 1
                if detoked:
                    block.append(fields[1].strip().replace("&quot;", "\"") + '\n')
                else:
                    block.append(md.detokenize(tok_target) + '\n')

                labels = []
                target_tokens = []
                while True:
                    fields = fin.readline().strip().split('\t')
                    row += 1

                    title = fields[0]
                    if title == 'Hullucination Score:':
                        # print(fields)
                        if len(fields) == 1:
                            break
                        score = fields[1].strip()
                        # print(target_tokens)
                        # print(tok_target)
                        # print(len(result_blocks['at']), len(result_blocks['mbart']))
                        assert bad_annot or len(target_tokens) == num_tokens

                        if score != '':
                            score = int(score)
                            if score == 2:
                                block.append(target_tokens)
                                block.append(labels)
                            elif score == 1:
                                block.append(target_tokens)
                                block.append([0 for _ in target_tokens])
                            elif score == 0:
                                block.append(target_tokens)
                                block.append([1 for _ in target_tokens])
                            else:
                                raise ValueError
                            total += 1
                            if bad_annot:
                                print(len(result_blocks), score, "score")
                            # print(target_tokens)
                            # print(tok_target)
                            if not detoked:
                                target_tokens, labels = detokenize_labels(target_tokens, block[-1])
                                block[-2] = target_tokens
                                block[-1] = labels
                            else:
                                labels = block[-1]
                            assert len(target_tokens) == len(labels)
                            block.append(score)
                            model, domain, idx = meta_data.readline().strip().split()
                            if not bad_annot:
                                result_blocks[model].append(block)
                        break
                    else:
                        tokens = fields[1:] if title == 'Word-separate Translation:' else fields
                        for ii, token in enumerate(tokens):
                            if token != '':
                                target_tokens.append(token.replace("&quot;", "\"") if detoked else token)
                                index = (row, ii+1)
                                labels.append(1 if index in activate_cells else 0)
                    assert len(labels) == len(target_tokens)

    print("In total {} annotations, bad = {}".format(total, total_bad_annotate))
    return result_blocks


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


def get_labels(annotations):
    return [annot[4] for annot in annotations], [annot[5] for annot in annotations]


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
    # print("PA = ", PA)

    # mean of squares of proportion of all assignments which were to jth category
    PE = sum([j ** 2 for j in [sum([rows[i] for rows in rate]) / (N * n) for i in range(k)]])
    # print("PE =", PE)

    kappa = -float("inf")
    try:
        kappa = (PA - PE) / (1 - PE)
        kappa = float("{:.3f}".format(kappa))
    except ZeroDivisionError:
        print("Expected agreement = 1")

    print("Fleiss' Kappa =", kappa)

    return kappa


def cal_token_level_kappa(labels_1, labels_2, labels_3):
    all_tokens = sum([len(ll) for ll in labels_1])
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


def cal_sent_level_kappa(labels_1, labels_2, labels_3):
    print("Sentence level fleiss-kappa")
    all_tokens = len([ll for ll in labels_1])
    k = 3
    n = 3
    mat = np.zeros((all_tokens, k))
    idx = 0
    for t1, t2, t3 in zip(labels_1, labels_2, labels_3):
        mat[idx, int(t1)] += 1
        mat[idx, int(t2)] += 1
        mat[idx, int(t3)] += 1
        idx += 1

    print(mat.shape)
    return fleissKappa(mat, n)


def cal_token_level_kappa_pair(labels_1, labels_2):
    all_tokens = sum([len(ll) for ll in labels_1])
    k = 2
    n = 2
    mat = np.zeros((all_tokens, k))
    idx = 0
    for l1, l2 in zip(labels_1, labels_2):
        for t1, t2 in zip(l1, l2):
            mat[idx, int(t1)] += 1
            mat[idx, int(t2)] += 1
            idx += 1
    return fleissKappa(mat, n)


def cal_sent_level_kappa_pair(labels_1, labels_2):
    print("Sentence level fleiss-kappa")
    all_tokens = len([ll for ll in labels_1])
    k = 3
    n = 2
    mat = np.zeros((all_tokens, k))
    idx = 0
    for t1, t2 in zip(labels_1, labels_2):
        mat[idx, int(t1)] += 1
        mat[idx, int(t2)] += 1
        idx += 1
    return fleissKappa(mat, n)


def aggregate(annotations):
    sources = [annot[0] for annot in annotations]
    refs = [annot[1] for annot in annotations]
    targets = [annot[2] for annot in annotations]
    target_tokens = [annot[3] for annot in annotations]
    labels = [annot[4] for annot in annotations]
    scores = [annot[5] for annot in annotations]

    agg_scores = np.zeros(4)
    for ss in scores:
        agg_scores[ss] += 1

    if agg_scores[0] >= 2:
        return sources[0], refs[0], targets[0], [-1]*len(labels[0]), [], \
               0, 0, len(labels[0]), len(labels[0])

    # print(target_tokens[0])
    # print(target_tokens[1])
    # print(target_tokens[2])
    # print(labels[0])
    # print(labels[1])
    # print(labels[2])
    # print(len(target_tokens[0]), len(target_tokens[1]), len(target_tokens[2]))
    assert len(labels[0]) == len(labels[1]) and len(labels[1]) == len(labels[2])
    assert len(target_tokens[0]) == len(target_tokens[1]) and len(target_tokens[1]) == len(target_tokens[2])
    assert len(labels[0]) == len(labels[1]) and len(labels[1]) == len(labels[2])

    word_counts = [0 for _ in range(len(target_tokens[0]))]
    for score, label in zip(scores, labels):
        for ii, ll in enumerate(label):
            word_counts[ii] += ll
    new_labels = [1 if cc >= 2 else 0 for cc in word_counts]
    agreement = [1 if cc == 3 or cc == 0 else 0 for cc in word_counts]
    hallucination_agree = [1 if cc > 1 else 0 for cc in word_counts]
    hallucination_total = [1 if cc > 0 else 0 for cc in word_counts]

    possible_labels = [1 if cc >= 1 else 0 for cc in word_counts]
    return sources[0], refs[0], targets[0], new_labels, possible_labels, \
           sum(hallucination_agree), sum(hallucination_total), sum(agreement), len(agreement)


root = "/Users/chuntinz/Documents/research/fairseq-hallucination/meta_eval/public_iter2"
meta_dir = "/Users/chuntinz/Desktop/latte_eval"
meta_path = os.path.join(meta_dir, "1_combine_at_mbart_round2_public_eval.metadata")

fout_sources, fout_targets, fout_refs, fout_labels, fout_possible_labels = [], [], [], [], []
for psrc, ptgt, pref, plabel, ppos_label in zip(fout_source_paths, fout_target_paths, fout_ref_paths, fout_label_paths, fout_possible_label_paths):
    fout_sources.append(open(psrc, "w", encoding="utf-8"))
    fout_targets.append(open(ptgt, "w", encoding="utf-8"))
    fout_refs.append(open(pref, "w", encoding="utf-8"))
    fout_labels.append(open(plabel, "w", encoding="utf-8"))
    fout_possible_labels.append(open(ppos_label, "w", encoding='utf-8'))

all_model_annots = {mtype:[] for mtype in model_types}
for worker_id in [1, 2, 3]:
    flabel = os.path.join(root, "public_worker_{}.label".format(worker_id))
    fsheet = os.path.join(root, "public_worker_{}.tsv".format(worker_id))
    annotate = read_sheet(flabel, fsheet, meta_path, detoked=True)
    for mtype in model_types:
        all_model_annots[mtype].append(annotate[mtype])

for mtype in model_types:
    print(mtype)
    tot = 0
    agree_words, tot_words = 0, 0
    ha_agree_words, ha_tot_words = 0, 0
    aggregate_labels = []
    aggregate_possible_labels = []

    model_idx = model_types.index(mtype)
    all_annots = all_model_annots[mtype]
    fout_source, fout_ref, fout_target, fout_label, fout_possible_label = fout_sources[model_idx], \
                                                                          fout_refs[model_idx], \
                                                                          fout_targets[model_idx], \
                                                                          fout_labels[model_idx], \
                                                                          fout_possible_labels[model_idx]
    for sid, (annot1, annot2, annot3) in enumerate(zip(*all_annots)):
        # print(sid)
        source, ref, target, labels, possible_labels, ha_agree_word_num, ha_word_num, \
        agree_word_num, word_num = aggregate([annot1, annot2, annot3])

        aggregate_labels.append(labels)

        agree_words += agree_word_num
        tot_words += word_num
        ha_agree_words += ha_agree_word_num
        ha_tot_words += ha_word_num

        if sum(labels) == -1 * len(labels):
            print("skip!")
            continue

        tot += 1
        fout_source.write(source.strip() + '\n')
        fout_ref.write(ref.strip() + '\n')
        fout_target.write(target.strip() + '\n')
        fout_label.write(" ".join([str(idx) for idx in labels]) + '\n')
        fout_possible_label.write(" ".join([str(idx) for idx in possible_labels]) + '\n')

        ttran = target.strip().replace("\"", "&quot;").split()
        segments = []
        words = 0
        seg = []
        seg_labels = []
        max_len_sheet = 11

        sent_labels = [annot1[5], annot2[5], annot3[5]]
        sent_label_inconsistency_scores = 0
        if len(set(sent_labels)) == 2:
            # print(sent_labels)
            sent_label_inconsistency_scores = (len(set(sent_labels)) * 3)
        elif len(set(sent_labels)) == 3:
            # print(sent_labels)
            sent_label_inconsistency_scores = (len(set(sent_labels)) * 100)

        word_label_inconsistency_scores = 0
        for ii in range(len(ttran)):
            seg.append(ttran[ii])
            word_labels = [annot1[4][ii], annot2[4][ii], annot3[4][ii]]
            if sum(word_labels) != 3 and sum(word_labels) != 0:
                word_label_inconsistency_scores += 1
                sent_label_inconsistency_scores += 1

            seg_labels.append(" ".join(list(map(str, [annot1[4][ii], annot2[4][ii], annot3[4][ii]]))))
            if (ii + 1) % max_len_sheet == 0 or ii == (len(ttran) - 1):
                words += len(seg)
                segments.append(" *" + "*".join(seg) + "\n" + " *" + "*".join(seg_labels))
                seg = []
                seg_labels = []

        assert words == len(ttran)

    print("Tot aggregation = {}, agree ratio = {}, hallucination agree ratio = {}".format(tot,
                                                                                          agree_words*1.0/tot_words,
                                                                                          ha_agree_words*1.0/ha_tot_words))

    fout_source.close()
    fout_target.close()
    fout_label.close()
    fout_possible_label.close()
    fout_ref.close()

    # calculate upper bound F1
    vendor1_labels, vendor1_scores = get_labels(all_annots[0])
    vendor2_labels, vendor2_scores = get_labels(all_annots[1])
    vendor3_labels, vendor3_scores = get_labels(all_annots[2])

    p1, r1, f1 = cal_f1(aggregate_labels, vendor1_labels)
    p2, r2, f2 = cal_f1(aggregate_labels, vendor2_labels)
    p3, r3, f3 = cal_f1(aggregate_labels, vendor3_labels)
    print("========== f1 of 3 annotators and mean ========== ", f1, f2, f3, np.mean([f1, f2, f3]))

    print(len(vendor3_scores))
    print("==================== fleiss kappa ===========================")
    cal_token_level_kappa(vendor1_labels, vendor2_labels, vendor3_labels)
    cal_sent_level_kappa(vendor1_scores, vendor2_scores, vendor3_scores)

    print("============== pairwise fleiss-kappa ========================")
    print(1, 2)
    cal_token_level_kappa_pair(vendor1_labels, vendor2_labels)
    cal_sent_level_kappa_pair(vendor1_scores, vendor2_scores)
    print(2, 3)
    cal_token_level_kappa_pair(vendor3_labels, vendor2_labels)
    cal_sent_level_kappa_pair(vendor3_scores, vendor2_scores)
    print(1, 3)
    cal_token_level_kappa_pair(vendor1_labels, vendor3_labels)
    cal_sent_level_kappa_pair(vendor1_scores, vendor3_scores)

    def check_score_dist(scores):
        dd = np.zeros(4)
        for ss in scores:
            dd[ss] += 1
        res = ["{:.3f}".format(score * 1.0 / sum(dd)) for score in dd]
        print(" ".join(res))

    check_score_dist(vendor1_scores)
    check_score_dist(vendor2_scores)
    check_score_dist(vendor3_scores)

    print("=="*50)