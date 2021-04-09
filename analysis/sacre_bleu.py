import sys
import sacrebleu

fsys = sys.argv[1]
fref = sys.argv[2]

refs = []
sys = []

with open(fref, "r", encoding="utf-8") as fin_ref, open(fsys, "r", encoding="utf-8") as fin_sys:
    for lref, lsys in zip(fin_ref, fin_sys):
        refs.append(lref.strip())
        sys.append(lsys.strip())

bleu = sacrebleu.corpus_bleu(sys, [refs])
print(bleu.score)
