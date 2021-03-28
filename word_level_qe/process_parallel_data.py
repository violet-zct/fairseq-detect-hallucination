import os
import shutil
import sentencepiece as spm
import sys
import numpy as np

root = "/home/chuntinz/tir5/data/qe_wmt18_ende"
opt_dir = os.path.join(root, "data2")
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)


model_path = "/home/chuntinz/tir5/pretrain_models/xlmr.large/"
vocab = os.path.join(model_path, 'sentencepiece.bpe.model')
sp = spm.SentencePieceProcessor()
sp.Load(vocab)


def spm(fin_path, fout_path):
    with open(fin_path, "r", encoding="utf-8") as fin, open(fout_path, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")


for split in ['corpus']:
    opt_split = 'para'

    en_fin = os.path.join(root, "others", "{}.en".format(split))
    de_fin = os.path.join(root, "others", "{}.de".format(split))

    en_sents = open(en_fin, "r", encoding="utf=8").readlines()
    de_sents = open(de_fin, "r", encoding="utf=8").readlines()

    K = 50000
    rand_array = np.random.permutation(list(range(len(en_sents))))

    with open(os.path.join(opt_dir, "{}.en".format(opt_split)), "w", encoding="utf=8") as fen, \
            open(os.path.join(opt_dir, "{}.ref".format(opt_split)), "w", encoding="utf=8") as fde:
        for ii in rand_array[:K]:
            fen.write(en_sents[ii])
            fde.write(de_sents[ii])

    spm(os.path.join(opt_dir, "{}.en".format(opt_split)), os.path.join(opt_dir, "{}.en.bpe".format(opt_split)))
    spm(os.path.join(opt_dir, "{}.ref".format(opt_split)), os.path.join(opt_dir, "{}.ref.bpe".format(opt_split)))