import os
import numpy as np
import sys
import random
import shutil

main_domain = sys.argv[1]
other_domain_ratio = 1

domains = ['law', "news", "patent", "tvsub"]
root_dir = "/private/home/chuntinz/work/data/multi-zh-en/detok/raw"
opt_dir = "/private/home/chuntinz/work/data/multi-zh-en/detok/raw/mix"

if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)


def convert_raw_labels_to_spm_labels(sent_spm, token_labels):
    sent_detoks = sent_spm.strip().replace(' ', '').replace('\u2581', ' ').split()
    token_labels = token_labels.strip().split()
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


def read_data(f1, f2):
    data = []
    with open(f1, encoding='utf-8') as fin1, open(f2, encoding='utf-8') as fin2:
        for l1, l2 in zip(fin1, fin2):
            data.append((l1.strip(), l2.strip()))
    return data


def main():
    train_data = dict()
    valid_data = dict()

    train_fsrc, train_ftgt = "train.en-zh.zh", "train.en-zh.en"
    valid_fsrc, valid_ftgt = "valid.en-zh.zh", "valid.en-zh.en"

    for d in domains:
        p1, p2 = os.path.join(root_dir, d, "train.en-zh.zh"), os.path.join(root_dir, d, "train.en-zh.en")
        train_data[d] = read_data(p1, p2)

        p1, p2 = os.path.join(root_dir, d, "valid.en-zh.zh"), os.path.join(root_dir, d, "valid.en-zh.en")
        valid_data[d] = read_data(p1, p2)

    total_data = sum([len(train_data[d]) for d in domains])
    domain_n = len(train_data[main_domain])
    random.shuffle(train_data[main_domain])

    ratios = [0.0001, 0.0003, 0.0005, 0.001, 1.0]
    ratios = [0.0003]

    mix_valid_data = []
    for d in domains:
        random.shuffle(valid_data[d])
        for l1, l2 in valid_data[d][:1000]:
            mix_valid_data.append((l1, l2))

    data = []
    for r in ratios:
        new_dir = os.path.join(opt_dir, f"mix-zhen-{main_domain}-{r}")
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        with open(os.path.join(new_dir, train_fsrc), 'w', encoding='utf-8') as fout1, \
                open(os.path.join(new_dir, train_ftgt), 'w', encoding='utf-8') as fout2:
            for d in domains:
                T = int(len(train_data[d]) * other_domain_ratio) if d != main_domain else int(r * domain_n)
                for l1, l2 in train_data[d][:T]:
                    data.append((l1, l2))
            random.shuffle(data)
            for l1, l2 in data:
                fout1.write(l1 + '\n')
                fout2.write(l2 + '\n')

        with open(os.path.join(new_dir, valid_fsrc), 'w', encoding='utf-8') as fout1, \
                open(os.path.join(new_dir, valid_ftgt), 'w', encoding='utf-8') as fout2:
            for l1, l2 in mix_valid_data:
                fout1.write(l1 + '\n')
                fout2.write(l2 + '\n')

        input_names = ['test.en-zh.en', 'test.en-zh.zh']
        output_names = ['test.en-zh.en', 'test.en-zh.zh']
        for fname, dname in zip(input_names, output_names):
            command = "cat {} > {}".format(" ".join([os.path.join(root_dir, d, fname) for d in domains if d != main_domain]),
                                       os.path.join(new_dir, "indomain." + dname))
            os.system(command)
            command = "cat {} > {}".format(os.path.join(root_dir, main_domain, fname),
                                       os.path.join(new_dir, main_domain + "." + dname))
            os.system(command)

main()
