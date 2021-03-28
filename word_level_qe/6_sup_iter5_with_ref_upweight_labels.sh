#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=20g
#SBATCH --cpus-per-task=5
##SBATCH --open-mode=append
#SBATCH --array=0

# please load your environment
source activate hal  #todo
# path to the XLM-R pretrained model
ROOT=/home/chuntinz/tir5/pretrain_models/xlmr.large #todo
# path to the synthetic data created with make_synthetic_data_mt.sh
datadir=/home/chuntinz/tir5/data/qe_wmt18_ende/data/bart_gen  #todo
DATABIN=${datadir}/mask_0.0_0.3_random_0.0_0.3_insert_0.2_wholeword_1_iters_5/bin  #todo
# path to the save directory
SAVE=checkpoints/unsup_iter5  #todo

rm -rf ${SAVE}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh
ln -s ${ROOT}/sentencepiece.bpe.model ${SAVE}/sentencepiece.bpe.model

TOTAL_NUM_UPDATES=300000  # Slightly change the hyperparameters for lr scheduling adapted from finetuning GLEU RTE task
WARMUP_UPDATES=18000
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=72        # Batch size.
MODEL_PATH=${ROOT}/model.pt  # Pretrained model path

# The following subsets are created under the binarized data by make_synthetic_data_mt.sh.
SRC=en  # subset name of source
TGT=de  # subset name of target
REF=ref  # subset name of reference

python -u train.py ${DATABIN}/ \
    --restore-file ${MODEL_PATH} \
    --task sentence_prediction --max-update 10000 --validate-interval-updates 500 \
    --input0 ${SRC} --input1 ${TGT} --input2 ${REF} \
    --add-ref-prob 1 --dropout-ref 0.7 \
    --add-tran-loss 1 --mask-prob 0.3 --masked-lm-loss-weight 0.3 \
    --add-target-num-tokens \
    --max-sentences ${MAX_SENTENCES} --max-tokens 4096 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion token_prediction \
    --num-classes ${NUM_CLASSES} \
    --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr ${LR} --total-num-update ${TOTAL_NUM_UPDATES} --warmup-updates ${WARMUP_UPDATES} \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --find-unused-parameters \
    --save-dir ${SAVE} --log-format simple \
    --skip-invalid-size-inputs-valid-test \
    --num-workers 0 --update-freq 1 \
    --no-epoch-checkpoints \
    --best-checkpoint-metric f1_mult --maximize-best-checkpoint-metric | tee ${SAVE}/log.txt
