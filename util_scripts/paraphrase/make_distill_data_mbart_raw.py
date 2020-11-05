import sys
import os
import numpy as np
from sacremoses import MosesTokenizer, MosesDetokenizer

# for training MT or NAT models
md = MosesDetokenizer(lang='en')
model_root = "/private/home/chuntinz/work/assets/fairseq/saved_models/"
raw_dir = "/private/home/chuntinz/work/data/multi-zh-en/detok/raw/mix/mix-zhen-patent-0.0003"

opt_root = "/private/home/chuntinz/work/data/distillation/multi-zh-en"

model_name = "21_mbart_zhen_mix_patent0.003"
log_name = "train_gen.log"
target_lang = "en"
source_lang = "zh" if target_lang == "en" else "en"

opt_dir = os.path.join(opt_root, model_name)
if not os.path.exists(opt_dir):
    os.makedirs(opt_dir)

# raw is used for bart generation
raw_opt_dir = os.path.join(opt_dir, "raw")

if not os.path.exists(raw_opt_dir):
    os.mkdir(raw_opt_dir)


def read_log(fname):
    hyp = []
    example_id = []
    with open(fname, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip().startswith("S-"):
                example_id.append(int(line.strip().split('\t')[0].split('-')[-1]))
            elif line.strip().startswith("D-"):
                hyp.append(line.strip().split('\t')[-1])
    assert len(example_id) == len(hyp)
    return hyp, example_id


def zh_bpe_to_raw(zh_bpe):
    tokenized = zh_bpe.strip().replace("@@ ", "")
    raw = tokenized.replace(" ", "").strip()
    return raw


def en_bpe_to_raw(en_bpe):
    raw = md.detokenize(en_bpe.strip().split()).strip()
    return raw


def read_raw_source_ref(example_ids):
    data = []
    with open(os.path.join(raw_dir, "train.en-zh.en"), 'r', encoding='utf-8') as fen, \
        open(os.path.join(raw_dir, "train.en-zh.zh"), 'r', encoding='utf-8') as fzh:
        for len, lzh in zip(fen, fzh):
            data.append((len.strip(), lzh.strip()))
    source = []
    ref = []

    ii = 0
    for idx in example_ids:
        source.append(data[idx][1])
        ref.append(data[idx][0])
        ii += 1
        if ii % 100000 == 0:
            print("processed {} lines!".format(ii))
    return source, ref


def write(zh_raw, en_raw, ref_raw,):
    random_indices = np.random.permutation(np.arange(len(zh_raw)))
    zh_raw = [zh_raw[idx] for idx in random_indices]
    en_raw = [en_raw[idx] for idx in random_indices]
    ref_raw = [ref_raw[idx] for idx in random_indices]

    if target_lang == "en":
        pair = zip(zh_raw, en_raw, ref_raw)
        fsrc = os.path.join(raw_opt_dir, "train.en-zh.zh")
        ftgt = os.path.join(raw_opt_dir, "train.en-zh.en")  # generation
        fref = os.path.join(raw_opt_dir, "train.en-zh.enref")
    else:
        pair = zip(en_raw, zh_raw, ref_raw)
        ftgt = os.path.join(raw_opt_dir, "train.en-zh.zh")
        fsrc = os.path.join(raw_opt_dir, "train.en-zh.en")
        fref = os.path.join(raw_opt_dir, "train.en-zh.zhref")

    with open(fsrc, 'w', encoding='utf-8') as fs, open(ftgt, 'w', encoding='utf-8') as ft, \
        open(fref, 'w', encoding='utf-8') as fr:
        for ls, lt, lr in pair:
            fs.write(ls + '\n')
            ft.write(lt + '\n')
            fr.write(lr.strip() + '\n')


hyp_bpe, example_ids = read_log(os.path.join(model_root, model_name, log_name))
hyp_raw = [en_bpe_to_raw(hyp) if target_lang == 'en' else zh_bpe_to_raw(hyp) for hyp in hyp_bpe]
# real data
zh_raw, en_raw = read_raw_source_ref(example_ids)

write(zh_raw, hyp_raw, en_raw)
