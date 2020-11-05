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
SRC=doc
TGT=sum

suffix=${1}  # the config in run_gen_synthetic_data_with_bart.sh
rootdir=${2}
input_dir="${rootdir}/bart_gen"
opt_root="${rootdir}/data"

# Path to parent directory of the GPT2 bpe (encoder.json, used by Roberta)
bpe_path=${3}
# Path to the dictionary used by the pretrained model (Roberta) for binarized data creation for finetuning
dict_path=${4}
iters=${5:-3}  # number of iterations in run_gen_synthetic_data_with_bart.sh

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
  bash util_scripts/preprocess/gpt2_encode_bpe.sh ${fname}.hypo ${fname}.hypo.bpe ${bpe_path}
  bash util_scripts/preprocess/gpt2_encode_bpe.sh ${fname}.ref ${fname}.ref.bpe ${bpe_path}

  python -u util_scripts/create_label_with_edit_distance.py ${fname}.ref.bpe ${fname}.hypo.bpe ${optdir}/${split}.label

  # convert bpe to vocab id for data binarization
  bash util_scripts/preprocess/gpt2_encode_ids.sh ${fname}.ref.bpe ${optdir}/${split}.ref ${bpe_path}
  bash util_scripts/preprocess/gpt2_encode_ids.sh ${fname}.hypo.bpe ${optdir}/${split}.${TGT} ${bpe_path}

  if [ ! -f ${rootdir}/${split}.${SRC}.ids ]; then
    bash util_scripts/preprocess/gpt2_encode_raw_to_ids.sh ${rootdir}/${split}.${SRC} ${rootdir}/${split}.${SRC}.ids ${bpe_path}
  fi

  if [ ${split} = "train" ]; then
    for (( i=1; i<=${iters}; i++ )); do cat ${rootdir}/${split}.${SRC}.ids >> ${optdir}/${split}.${SRC}.original; done
  else
    cp ${rootdir}/${split}.${SRC}.ids ${optdir}/${split}.${SRC}.original
  fi

  # truncate document since the maximum position allowed by Roberta is 512 (this might be an issue, as we observe the truncation length is relatively high)
  python util_scripts/preprocess/truncate_sentence_xsum.py ${optdir}/${split}.${SRC}.original \
    ${optdir}/${split}.${TGT} ${optdir}/${split}.ref ${optdir}/${split}.${SRC}
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
fairseq-preprocess --only-source --trainpref ${inputdir}/train.${split} --destdir ${optdir}/${split} \
  --validpref ${inputdir}/valid.${split} --workers 30
