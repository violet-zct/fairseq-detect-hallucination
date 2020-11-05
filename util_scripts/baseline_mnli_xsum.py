import torch
from fairseq.data.data_utils import collate_tokens
import os
from scipy import stats
import numpy as np
from fairseq.models.roberta import RobertaModel

modelroot = '/private/home/chuntinz/work/fairseq-hallucination/container'
model = 'roberta.large.mnli'

roberta = RobertaModel.from_pretrained(
    '{}/{}/'.format(modelroot, model),
    checkpoint_file='model.pt',
    data_name_or_path='{}/{}/'.format(modelroot, model)
)
print('Loads the model!')

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
roberta.half()

test_dirs = ["evals/xsum/PtGen", "evals/xsum/TConvS2S", "evals/xsum/TranS2S", "evals/xsum/BERTS2S"]
test_prefix = ["PtGen", "TConvS2S", "TranS2S", "BERTS2S"]

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
            # prediction = roberta.predict('mnli', tokens)[0, 2].item()
            # predict_hallucination_strengths_by_probs.append(1 - np.exp(prediction))
            prediction = roberta.predict('mnli', tokens)[0, 0].item()
            predict_hallucination_strengths_by_probs.append(np.exp(prediction))
            nsamples += 1
    spearman_corr_by_probs, p_value_by_probs = stats.spearmanr(gold_hallucination_strengths,
                                                               predict_hallucination_strengths_by_probs)
    print('Spearman-corr by probs: {}'.format(spearman_corr_by_probs))