import sys
import os
import numpy as np
from sacremoses import MosesTokenizer, MosesDetokenizer

# for training MT or NAT models
md = MosesDetokenizer(lang='en')
model_root = "/private/home/chuntinz/work/assets/fairseq-domain-shift/saved_ha_models"
bpe_dir = "/checkpoint/chuntinz/data/multi-zh-en/detok/mix-data-bpe/mix-zhen-patent-0.0003"

opt_root = "/private/home/chuntinz/work/data/distillation/multi-zh-en"

model_name = "26_v2_mt_mix_patent_0.0003"
log_name = "train.zh.tran.log"
target_lang = "en"
source_lang = "zh" if target_lang == "en" else "en"

opt_dir = os.path.join(opt_root, model_name)
if not os.path.exists(opt_dir):
    os.makedirs(opt_dir)

# raw is used for bart generation
raw_opt_dir = os.path.join(opt_dir, "raw")
# bpe is used for (1) train NAT (2) back-translation
bpe_opt_dir = os.path.join(opt_dir, "bpe")

if not os.path.exists(raw_opt_dir):
    os.mkdir(raw_opt_dir)
if not os.path.exists(bpe_opt_dir):
    os.mkdir(bpe_opt_dir)


def read_log(fname):
    hyp = []
    example_id = []
    with open(fname, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line.strip().startswith("S-"):
                example_id.append(int(line.strip().split('\t')[0].split('-')[-1]))
            elif line.strip().startswith("H-"):
                hyp.append(line.strip().split('\t')[-1])
    assert len(example_id) == len(hyp)
    return hyp, example_id


def zh_bpe_to_raw(zh_bpe):
    tokenized = zh_bpe.strip().replace("@@ ", "")
    raw = tokenized.replace(" ", "").strip()
    return raw


def en_bpe_to_raw(en_bpe):
    tokenized = en_bpe.replace("@@ ", "").strip().split()
    raw = md.detokenize(tokenized).strip()
    return raw


def read_raw_source_ref(example_ids):
    data = []
    bpe_data = []
    with open(os.path.join(bpe_dir, "train.en-zh.en"), 'r', encoding='utf-8') as fen, \
        open(os.path.join(bpe_dir, "train.en-zh.zh"), 'r', encoding='utf-8') as fzh:
        for len, lzh in zip(fen, fzh):
            bpe_data.append((len.strip(), lzh.strip()))
            ren, rzh = en_bpe_to_raw(len), zh_bpe_to_raw(lzh)
            data.append((ren, rzh))
    source = []
    ref = []

    bpe_source, bpe_ref = [], []
    ii = 0
    for idx in example_ids:
        source.append(data[idx][1])
        ref.append(data[idx][0])
        bpe_source.append(bpe_data[idx][1])
        bpe_ref.append(bpe_data[idx][0])
        ii += 1
        if ii % 100000 == 0:
            print("processed {} lines!".format(ii))
    return source, ref, bpe_source, bpe_ref


def test():
    root = "/checkpoint/chuntinz/data/multi-zh-en/detok/bpe/law"
    opt = "/checkpoint/chuntinz/data/multi-zh-en/detok/bpe/law/temp"
    with open(os.path.join(root, "train.en-zh.en"), 'r', encoding='utf-8') as fen, \
            open(os.path.join(root, "train.en-zh.zh"), 'r', encoding='utf-8') as fzh, \
            open(os.path.join(opt, "en"), 'w', encoding='utf-8') as en_out, \
            open(os.path.join(opt, "zh"), 'w', encoding='utf-8') as zh_out:
        for len, lzh in zip(fen, fzh):
            ren, rzh = en_bpe_to_raw(len), zh_bpe_to_raw(lzh)
            en_out.write(ren + '\n')
            zh_out.write(rzh + '\n')


def write(zh_raw, en_raw, ref_raw, source_bpe):
    if target_lang == "en":
        pair = zip(zh_raw, en_raw, ref_raw, source_bpe)
        fsrc = os.path.join(raw_opt_dir, "train.en-zh.zh")
        ftgt = os.path.join(raw_opt_dir, "train.en-zh.en")  # generation
        fsrc_bpe = os.path.join(bpe_opt_dir, "train.en-zh.zh")
        fref = os.path.join(raw_opt_dir, "train.en-zh.enref")
    else:
        pair = zip(en_raw, zh_raw, ref_raw, source_bpe)
        ftgt = os.path.join(raw_opt_dir, "train.en-zh.zh")
        fsrc = os.path.join(raw_opt_dir, "train.en-zh.en")
        fsrc_bpe = os.path.join(bpe_opt_dir, "train.en-zh.en")
        fref = os.path.join(raw_opt_dir, "train.en-zh.zhref")

    with open(fsrc, 'w', encoding='utf-8') as fs, open(ftgt, 'w', encoding='utf-8') as ft, \
        open(fref, 'w', encoding='utf-8') as fr, open(fsrc_bpe, 'w', encoding='utf-8') as fs_bpe:
        for ls, lt, lr, ls_bpe in pair:
            fs.write(ls + '\n')
            ft.write(lt + '\n')
            fr.write(lr.strip() + '\n')
            fs_bpe.write(ls_bpe.strip() + '\n')


hyp_bpe, example_ids = read_log(os.path.join(model_root, model_name, log_name))
hyp_raw = [en_bpe_to_raw(hyp) if target_lang == 'en' else zh_bpe_to_raw(hyp) for hyp in hyp_bpe]
# real data
zh_raw, en_raw, zh_bpe, en_bpe = read_raw_source_ref(example_ids)

write(zh_raw, hyp_raw, en_raw, en_bpe if target_lang == 'zh' else zh_bpe)
