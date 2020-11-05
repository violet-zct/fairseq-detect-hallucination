import torch
from fairseq.data.data_utils import collate_tokens
import os
from scipy import stats
import numpy as np
from fairseq.models.roberta import RobertaModel

modelroot = '/private/home/chuntinz/work/fairseq-hallucination/checkpoints'
model = 'baseline_xnli_finetune_xlmr'

roberta = RobertaModel.from_pretrained(
    '{}/{}/'.format(modelroot, model),
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/private/home/chuntinz/work/data/xnli/zhen_bin/'
)
print('Loads the model!')

print(roberta.task._label_dictionary.indices)
label_map = {"contradictory": 0, "entailment": 1, "neutral": 2}
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
roberta.half()

test_dirs = ["evals/public_round1_v3_final"]
test_prefix = ["v2.r1"]

for prefix, test_dir in zip(test_prefix, test_dirs):
    print(prefix)
    predict_hallucination_strengths_by_probs = []
    gold_hallucination_strengths = []

    with open(os.path.join(test_dir, prefix + ".source"), encoding='utf-8') as fsrc, \
            open(os.path.join(test_dir, prefix + ".target"), encoding='utf-8') as ftgt, \
            open(os.path.join(test_dir, prefix+".label"), encoding='utf-8') as flabel:
        for src, tgt, label in zip(fsrc, ftgt, flabel):
            gold_strengths = sum([int(l) for l in label.strip().split()]) * 1.0 / len(label.strip().split())
            gold_hallucination_strengths.append(gold_strengths)
            tokens, _, _ = roberta.encode(src, tgt)
            # entailment_prediction = roberta.predict('sentence_classification_head', tokens)[0, 1].item()
            # predict_hallucination_strengths_by_probs.append(1 - np.exp(entailment_prediction))
            contradict_prediction = roberta.predict('sentence_classification_head', tokens)[0, 0].item()
            predict_hallucination_strengths_by_probs.append(np.exp(contradict_prediction))
            nsamples += 1
    spearman_corr_by_probs, p_value_by_probs = stats.spearmanr(gold_hallucination_strengths,
                                                               predict_hallucination_strengths_by_probs)
    print('Spearman-corr by probs: {}'.format(spearman_corr_by_probs))