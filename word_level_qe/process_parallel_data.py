import os
import shutil
import sentencepiece as spm
import sys


root = "/home/chuntinz/tir5/data/qe_wmt18_ende"
opt_dir = os.path.join(root, "data2")
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)


model_path = "/home/chuntinz/tir5/pretrain_models/mbart.cc100"
vocab = os.path.join(model_path, 'sentence.bpe.model')
sp = spm.SentencePieceProcessor()
sp.Load(vocab)


def spm(fin_path, fout_path):
    with open(fin_path, "r", encoding="utf-8") as fin, open(fout_path, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")


for split in ['corpus']:
    opt_split = 'para'

    shutil.copy(os.path.join(root, split, "{}.en".format(split)), os.path.join(opt_dir, "{}.en".format(opt_split)))
    shutil.copy(os.path.join(root, split, "{}.de".format(split)), os.path.join(opt_dir, "{}.ref".format(opt_split)))

    spm(os.path.join(opt_dir, "{}.en".format(opt_split)), os.path.join(opt_dir, "{}.en.bpe".format(opt_split)))
    spm(os.path.join(opt_dir, "{}.ref".format(opt_split)), os.path.join(opt_dir, "{}.ref.bpe".format(opt_split)))