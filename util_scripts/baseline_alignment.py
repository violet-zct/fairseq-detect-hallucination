from simalign import SentenceAligner
import sys
import os
import numpy as np
from scipy import stats


option = sys.argv[1]
eval_dir = "/private/home/chuntinz/work/fairseq-hallucination/evals"
if option == "1":
    print("MT")
    model = "xlm-roberta-large"
    eval_dir = os.path.join(eval_dir, "public_round1_v3_final")
    prefix = ["v2.r1"]
    unfold_dir = False
else:
    print("XSum")
    model = "roberta-base"
    eval_dir = os.path.join(eval_dir, "xsum")
    prefix = ["PtGen", "TConvS2S", "TranS2S", "BERTS2S"]
    unfold_dir = True

# making an instance of our model.
# You can specify the embedding model and all alignment settings in the constructor.
myaligner = SentenceAligner(model=model, token_type="word", matching_methods="mai")

for pp in prefix:
    print(pp)
    if unfold_dir:
        src_path = os.path.join(eval_dir, pp, pp+".source")
        tran_path = os.path.join(eval_dir, pp, pp+".target")
        label_path = os.path.join(eval_dir, pp, pp+".label")
    else:
        src_path = os.path.join(eval_dir, pp + ".source.tok")
        tran_path = os.path.join(eval_dir, pp+".target")
        label_path = os.path.join(eval_dir, pp+".label")

    gold_strengths = []
    prediction_strengths_by_aligned_token = []
    prediction_strengths_by_aligned_simla = []
    cc = 0
    with open(src_path, "r", encoding="utf-8") as fsrc, open(tran_path, "r", encoding="utf-8") as ftgt, \
        open(label_path, "r", encoding="utf-8") as flabel:
        for lsrc, ltgt, llabel in zip(fsrc, ftgt, flabel):
            src, tgt, label = lsrc.strip().split(), ltgt.strip().split(), llabel.strip().split()
            label = [float(l) for l in label]
            gold_strengths.append(sum(label) * 1.0 / len(label))

            alignments, sim_mat = myaligner.get_word_aligns(src, tgt)
            itermax_alignments = alignments['itermax']

            avg_max_aligned_src_sim = np.mean(np.max(sim_mat, axis=0))
            prediction_strengths_by_aligned_simla.append(1 - avg_max_aligned_src_sim)

            ratio_words_aligned_in_target = len([b for a, b in itermax_alignments]) * 1.0 / len(tgt)
            prediction_strengths_by_aligned_token.append(1 - ratio_words_aligned_in_target)

            cc += 1
            if cc % 50 == 0:
                print("Processed {} lines!".format(cc))

        spearman_corr_by_probs, p_value_by_probs = stats.spearmanr(gold_strengths,
                                                                   prediction_strengths_by_aligned_simla)
        spearman_corr_by_token, p_value_by_token = stats.spearmanr(gold_strengths,
                                                                   prediction_strengths_by_aligned_token)
        print('Spearman-corr by similarity: {}'.format(spearman_corr_by_probs))
        print('Spearman-corr by ratios of aligned target tokens: {}'.format(spearman_corr_by_token))