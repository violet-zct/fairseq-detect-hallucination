import torch
from fairseq.models.bart import BARTModel
import os

bart = BARTModel.from_pretrained(
    '/private/home/chuntinz/work/fairseq-hallucination/checkpoints/26_finetune_bart_xsum',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/private/home/chuntinz/work/data/summarization/raw/gpt2_bped/bin'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32

root = "/private/home/chuntinz/work/data/summarization/raw/bart_distill"
inputs = [os.path.join(root, 'train.document'), os.path.join(root, 'valid.document')]
outputs = [os.path.join(root, 'train.summary'), os.path.join(root, 'valid.summary')]

for fsrc, ftgt in zip(inputs, outputs):
    with open(fsrc, 'r', encoding='utf-8') as source, open(ftgt, 'w', encoding='utf-8') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch, _ = bart.sample(slines, beam=6, lenpen=1.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch, _ = bart.sample(slines, beam=6, lenpen=1.0, max_len_b=60, min_len=10, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()