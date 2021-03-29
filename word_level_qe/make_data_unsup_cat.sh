#!/bin/bash

# The directory is organized as follows:
# - rootdir (contains the original raw bi-text data)
#   - train.${SRC}
#   - train.${TGT}
#   - valid.${SRC}
#   - valid.${TGT}
#   - bart_gen
#     - xxx.hypo.bpe
#     - xxx.ref.bpe
#   - data
#     - directory of new synthetic dataset
source activate hal

# Modify this as needed
SRC=en
TGT=de

suffix1=${1}  # the config in run_gen_synthetic_data_with_bart.sh
suffix2=${2}
rootdir=/home/chuntinz/tir5/data/qe_wmt18_ende/data2 #${2}  #todo
opt_root="${rootdir}/bart_gen"

# Path to the sentencepiece model (used by xlmr)
# Path to the dictionary used by the pretrained model (XLM-R) for binarized data creation for finetuning
dict_path=/home/chuntinz/tir5/pretrain_models/xlmr.large/dict.txt #${4}  #todo
#iters=${5:-3}  # number of iterations in run_gen_synthetic_data_with_bart.sh

## If you are finetuning with Roberta (English task) or XLM-Roberta (crosslingual task), set the corresponding model here
#finetune_model=xlmr

optdir=${opt_root}/combine_unsup
unsup_dir_1=${opt_root}/${suffix1}
unsup_dir_2=${opt_root}/${suffix2}
rm -rf ${optdir}
mkdir -p ${optdir}

cat $unsup_dir_2/train.label $unsup_dir_1/train.label > $optdir/train.label
cat $unsup_dir_2/train.${TGT} $unsup_dir_1/train.${TGT} > $optdir/train.${TGT}
cat $unsup_dir_2/train.${SRC} $unsup_dir_1/train.${SRC} > $optdir/train.${SRC}
cat $unsup_dir_2/train.ref $unsup_dir_1/train.ref > $optdir/train.ref

cp $rootdir/valid.bpe.labels $optdir/valid.label
cp $rootdir/valid.de.bpe $optdir/valid.${TGT}
cp $rootdir/valid.ref.bpe $optdir/valid.ref
cp $rootdir/valid.en.bpe $optdir/valid.${SRC}

# create binarized data for fairseq
inputdir=${optdir}
optdir=${optdir}/bin

for split in ${SRC} ${TGT} ref; do
fairseq-preprocess --only-source --trainpref ${inputdir}/train.${split} --validpref ${inputdir}/valid.${split} \
--destdir ${optdir}/${split} --workers 30 --srcdict ${dict_path}
done

# binarize the label set
split=label
fairseq-preprocess --only-source --trainpref ${inputdir}/train.${split} --validpref ${inputdir}/valid.${split} \
--destdir ${optdir}/${split} --workers 30