#! /bin/bash
#!/bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=mbart
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=30g
#SBATCH --time=4320
#SBATCH --array=0

source activate hal

# Tune these hyperparameters for synthetic noises
low_mask_prob=0.0
high_mask_prob=0.5
low_random_prob=0.0
high_random_prob=0.3
insert_prob=0.2
mask_length="none"
random_word_span=0
mask_whole_word=1
# If the target language is not English, you could also generate synthetic data with mbart.
gen_with_mbart=1
iters=5  # generate this many number of noised targets for each input line

# Fill your input file path (the target data in the bitext training corpus) and output directory
input="/home/chuntinz/tir5/data/qe_wmt18_ende/data/train.de"
opt_dir=$(dirname "${input}")/bart_gen
if [ ! -d ${opt_dir} ]; then
  mkdir -p ${opt_dir}
fi

config="mask_${low_mask_prob}_${high_mask_prob}_random_${low_random_prob}_${high_random_prob}_insert_${insert_prob}_wholeword_${mask_whole_word}"
train_output=${opt_dir}/train.${config}

python -u util_scripts/gen_bart_batch.py \
  --gen-with-mbart ${gen_with_mbart} \
  --iters ${iters} \
  --input ${input} \
  --output ${train_output} \
  --low-mask-prob ${low_mask_prob} \
  --high-mask-prob ${high_mask_prob} \
  --low-random-prob ${low_random_prob} \
  --high-random-prob ${high_random_prob} \
  --mask-length ${mask_length} \
  --random-word-span ${random_word_span} \
  --insert-ratio ${insert_prob} \
  --replace-length 1 \
  --poisson-lambda 3.5 \
  --mask_whole_word ${mask_whole_word}