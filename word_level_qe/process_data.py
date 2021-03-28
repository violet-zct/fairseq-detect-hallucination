import os
import shutil
import sentencepiece as spm
import sys


root = "/home/chuntinz/tir5/data/qe_wmt18_ende"
opt_dir = os.path.join(root, "data2")
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)


model_path = "/home/chuntinz/tir5/pretrain_models/xlmr.large/"
vocab = os.path.join(model_path, 'sentencepiece.bpe.model')
sp = spm.SentencePieceProcessor()
sp.Load(vocab)


def convert_raw_labels_to_spm_labels(sent_spm, token_labels):
    sent_detoks = sent_spm.strip().replace(' ', '').replace('\u2581', ' ').split()
    assert len(sent_detoks) == len(token_labels)

    new_labels = []
    atom = []
    detok_pointer = 0
    for spm in sent_spm.strip().split():
        if len(atom) == 0:
            atom.append(spm)
        elif " ".join(atom).replace(' ', '').replace('\u2581', ' ').strip() == sent_detoks[detok_pointer]:
            new_labels.extend([token_labels[detok_pointer] for _ in atom])
            detok_pointer += 1

            atom = []
            atom.append(spm)
        else:
            atom.append(spm)
    if len(atom) > 0:
        assert " ".join(atom).replace(' ', '').replace('\u2581', ' ').strip() == sent_detoks[detok_pointer]
        new_labels.extend([token_labels[detok_pointer] for _ in atom])
    assert len(new_labels) == len(sent_spm.strip().split())
    assert detok_pointer == len(sent_detoks) - 1
    return " ".join(new_labels)


def spm(fin_path, fout_path):
    with open(fin_path, "r", encoding="utf-8") as fin, open(fout_path, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")


def spm_labels(fin_path, fout_path, tags, label_path):
    with open(fin_path, "r", encoding="utf-8") as fin, open(fout_path, "w", encoding="utf-8") as fout, \
            open(label_path, "w", encoding="utf-8") as flabel:
        for line, tok_tag in zip(fin, tags):
            spm_sent = " ".join(sp.EncodeAsPieces(line.strip())) + "\n"
            convert_tags = convert_raw_labels_to_spm_labels(spm_sent, tok_tag)
            fout.write(spm_sent)
            flabel.write(convert_tags + '\n')


for split in ['train', 'dev', 'test']:
    if split == 'dev':
        opt_split = "valid"
    else:
        opt_split = split

    shutil.copy(os.path.join(root, split, "{}.src".format(split)), os.path.join(opt_dir, "{}.en".format(opt_split)))
    shutil.copy(os.path.join(root, split, "{}.mt".format(split)), os.path.join(opt_dir, "{}.de".format(opt_split)))
    shutil.copy(os.path.join(root, split, "{}.pe".format(split)), os.path.join(opt_dir, "{}.ref".format(opt_split)))

    tags_list = []
    with open(os.path.join(root, split, '{}.tags'.format(split)), "r") as fin, open(os.path.join(opt_dir, '{}.tok.labels'.format(opt_split)), "w") as fout:
        for line in fin:
            fields = line.strip().split()
            assert len(fields) % 2 == 1
            tags = ["0" if tt == 'OK' else '1' for tt in fields[1::2]]
            tags_list.append(tags)
            fout.write(" ".join(tags) + '\n')

    if split == 'test':
        continue

    spm(os.path.join(opt_dir, "{}.en".format(opt_split)), os.path.join(opt_dir, "{}.en.bpe".format(opt_split)))
    spm_labels(os.path.join(opt_dir, "{}.de".format(opt_split)), os.path.join(opt_dir, "{}.de.bpe".format(opt_split)),
               tags_list, os.path.join(opt_dir, '{}.bpe.labels'.format(opt_split)))
    spm(os.path.join(opt_dir, "{}.ref".format(opt_split)), os.path.join(opt_dir, "{}.ref.bpe".format(opt_split)))