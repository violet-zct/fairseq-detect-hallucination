#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=100g
#SBATCH --cpus-per-task=10
##SBATCH --open-mode=append
#SBATCH --array=0

# please load your environment
source activate py37
# path to the Roberata pretrained model
ROOT=/private/home/chuntinz/work/fairseq-hallucination/pretrain_scripts/container/roberta.large
# path to the synthetic data created with make_synthetic_data_mt.sh
DATABIN=$1
# path to the save directory
SAVE=checkpoints/finetune_xsum

cp $0 ${SAVE}/run.sh

TOTAL_NUM_UPDATES=300000  # Slightly change the hyperparameters for lr scheduling adapted from finetuning GLEU RTE task
WARMUP_UPDATES=18000
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=6        # Batch size.
MODEL_PATH=${ROOT}/model.pt  # Pretrained model path

# The following subsets are created under the binarized data by make_synthetic_data_mt.sh.
SRC=doc  # subset name of source
TGT=sum   # subset name of target
REF=ref  # subset name of reference

python -u train.py ${DATABIN}/ \
    --restore-file ${MODEL_PATH} \
    --task sentence_prediction \
    --max-update 16000 --input0 ${SRC} --input1 ${TGT} --input2 ${REF} \
    --add-ref-prob 1 --dropout-ref 0.5 \
    --add-tran-loss 1 --mask-prob 0.3 --masked-lm-loss-weight 0.5 \
    --add-target-num-tokens \
    --max-sentences ${MAX_SENTENCES} --max-tokens 4096 \
    --num-workers 0 --update-freq 4 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion token_prediction \
    --num-classes $NUM_CLASSES \
    --max-positions 512 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --find-unused-parameters \
    --save-dir ${SAVE} --log-format simple \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints --keep-interval-updates 6 --save-interval-updates 2000 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric | tee ${SAVE}/log.txt