#!/bin/bash

source activate py37

model_name="26_v2_mt_mix_patent_0.0003"

raw_dir="/private/home/chuntinz/work/data/distillation/multi-zh-en/${model_name}/raw"
bpe_dir="/private/home/chuntinz/work/data/distillation/multi-zh-en/${model_name}/bpe"

enbpe="/private/home/chuntinz/work/data/multi-zh-en/detok/tok/en.bpe"
zhbpe="/private/home/chuntinz/work/data/multi-zh-en/detok/tok/zh.bpe"

python make_raw_data.py

subword-nmt apply-bpe -c ${enbpe} < ${raw_dir}/train.en-zh.en > ${bpe_dir}/train.en-zh.en

root=/checkpoint/chuntinz/data/bin/multi-zhen-v1/mix
tdd=${root}/all
slang=en
tlang=zh
edpath=${tdd}/dict.en.txt
ddpath=${tdd}/dict.zh.txt

cd ../../
python preprocess.py --srcdict $edpath --tgtdict $ddpath --source-lang ${slang} --target-lang ${tlang} \
 --workers 30 --trainpref ${bpe_dir}/train.en-zh --destdir ${bpe_dir}/bin

cp /private/home/chuntinz/work/data/bin/multi-zhen-v1/mix/mix-zhen-patent-0.0003/test* ${bpe_dir}/bin
cp /private/home/chuntinz/work/data/bin/multi-zhen-v1/mix/mix-zhen-patent-0.0003/valid* ${bpe_dir}/bin