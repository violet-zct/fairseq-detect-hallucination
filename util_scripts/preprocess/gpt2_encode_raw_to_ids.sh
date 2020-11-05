#!/bin/bash

#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
#wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

source activate py37

INPUT=$1
OUTPUT=$2
BPEPATH=${3:-/private/home/chuntinz/work/fairseq-hallucination/container/gpt2_bpe}

python -m examples.roberta.multiprocessing_bpe_encoder \
--encoder-json ${BPEPATH}/encoder.json \
--vocab-bpe ${BPEPATH}/vocab.bpe \
--inputs ${INPUT} \
--outputs ${OUTPUT} \
--workers 5 \
--keep-empty