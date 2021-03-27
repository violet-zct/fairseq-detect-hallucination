import os
import shutil


root = "/home/chuntinz/tir5/data/qe_wmt18_ende"

opt_dir = os.path.join(root, "data")

if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)

for split in ['train', 'dev', 'test']:
    if split == 'dev':
        opt_split = "valid"
    else:
        opt_split = split

    shutil.copy(os.path.join(root, split, "{}.src".format(split)), os.path.join(opt_dir, "{}.en".format(opt_split)))
    shutil.copy(os.path.join(root, split, "{}.mt".format(split)), os.path.join(opt_dir, "{}.tran".format(opt_split)))
    shutil.copy(os.path.join(root, split, "{}.pe".format(split)), os.path.join(opt_dir, "{}.de".format(opt_split)))

    with open(os.path.join(root, split, '{}.tags'.format(split)), "r") as fin, open(os.path.join(opt_dir, '{}.labels'.format(opt_split)), "w") as fout:
        for line in fin:
            fields = line.strip().split()
            assert len(fields) % 2 == 1
            tags = ["0" if tt == 'OK' else '1' for tt in fields[1::2]]
            fout.write(" ".join(tags) + '\n')