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

# Modify this as needed
SRC=zh
TGT=en

suffix=${1}  # the config in run_gen_synthetic_data_with_bart.sh
rootdir=${2}
input_dir="${rootdir}/bart_gen"
opt_root="${rootdir}/data"

# Path to the sentencepiece model (used by xlmr)
bpe_path=${3}
# Path to the dictionary used by the pretrained model (XLM-R) for binarized data creation for finetuning
dict_path=${4}
iters=${5:-3}  # number of iterations in run_gen_synthetic_data_with_bart.sh

## If you are finetuning with Roberta (English task) or XLM-Roberta (crosslingual task), set the corresponding model here
#finetune_model=xlmr

optdir=${opt_root}/${suffix}
if [ ! -d ${optdir} ]; then
  mkdir -p ${optdir}
fi

for split in valid train; do
  # Path to the synthetic target generated with run_gen_synthetic_data_with_bart.sh
  fname=${input_dir}/${split}.${suffix}
  # linux grep may have problems in certain cases, we use the following grep.py to grep the synthetic target and reference from the log file
  # Usage: grep.py input output1 output2
  python util_scripts/grep.py ${fname} ${fname}.hypo ${fname}.ref

  # After the following executions, the bpes (used for final finetuning) of synthetic target and reference target will be generated,
  # and the pseudo labels will be created on synthetic target with edit-distance between them.
  python util_scripts/spm_encode.py ${fname}.hypo ${bpe_path}
  python util_scripts/spm_encode.py ${fname}.ref ${bpe_path}

  python -u util_scripts/create_label_with_edit_distance.py ${fname}.ref.bpe ${fname}.hypo.bpe ${optdir}/${split}.label

  # NOTE: if you used distillation for synthetic data creation, then the xx.ref is the distillation data,
  # please modify to the true reference accordingly (don't forget to bpe first)
  mv ${fname}.ref.bpe ${optdir}/${split}.ref
  mv ${fname}.hypo.bpe ${optdir}/${split}.${TGT}
  if [ ! -f ${rootdir}/${split}.${SRC}.bpe ]; then
    python util_scripts/spm_encode.py ${rootdir}/${split}.${SRC} ${bpe_path}
  fi

  if [ ${split} = "train" ]; then
    for (( i=1; i<=${iters}; i++ )); do cat ${rootdir}/${split}.${SRC}.bpe >> ${optdir}/${split}.${SRC}; done
  else
    cp ${rootdir}/${split}.${SRC}.bpe ${optdir}/${split}.${SRC}
  fi
done

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