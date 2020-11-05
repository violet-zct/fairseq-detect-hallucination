#!/bin/bash

raw_dir="/private/home/chuntinz/work/data/xnli/zh_en_tok"
spm_dir="/private/home/chuntinz/work/data/xnli/zh_en_spm"

for split in train valid test; do
  python /private/home/chuntinz/work/fairseq-hallucination/pretrain_scripts/spm_encode.py ${raw_dir}/${split}.sent1 ${spm_dir}/${split}.sent1
  python /private/home/chuntinz/work/fairseq-hallucination/pretrain_scripts/spm_encode.py ${raw_dir}/${split}.sent2 ${spm_dir}/${split}.sent2
  cp ${raw_dir}/${split}.label ${spm_dir}/${split}.label
done

optdir="/private/home/chuntinz/work/data/xnli/zhen_bin"
dict_path=/private/home/chuntinz/work/fairseq-hallucination/pretrain_scripts/container/xlmr.large
for split in sent1 sent2; do
  fairseq-preprocess --only-source --trainpref ${spm_dir}/train.${split} --validpref ${spm_dir}/valid.${split} \
--testpref ${spm_dir}/test.${split} --destdir ${optdir}/${split} --workers 40 --srcdict ${dict_path}/dict.txt
done

split=label
fairseq-preprocess --only-source --trainpref ${spm_dir}/train.${split} --validpref ${spm_dir}/valid.${split} \
--testpref ${spm_dir}/test.${split} --destdir ${optdir}/${split} --workers 40
