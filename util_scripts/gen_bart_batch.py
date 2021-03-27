import torch
from fairseq.models.bart import BARTModel
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=1, help="generate this many number of noised targets for each input line")
parser.add_argument("--input", type=str, default=None)
parser.add_argument("--output", type=str, default=None)

parser.add_argument("--mask-length", type=str, default="span-poisson")
parser.add_argument("--mask-ratio", type=float, default=0.4)
parser.add_argument("--random-ratio", type=float, default=0.0)
parser.add_argument("--poisson-lambda", type=float, default=3.5)
parser.add_argument("--insert-ratio", type=float, default=0.2)
parser.add_argument("--replace-length", type=int, default=1)
parser.add_argument("--mask_whole_word", type=int, default=1)
parser.add_argument("--high-noise-prob", type=float, default=0.)
parser.add_argument("--low-noise-prob", type=float, default=0.)

parser.add_argument("--low-mask-prob", type=float, default=0.)
parser.add_argument("--high-mask-prob", type=float, default=0.8)
parser.add_argument("--low-random-prob", type=float, default=0.0)
parser.add_argument("--high-random-prob", type=float, default=0.4)

parser.add_argument("--random-word-span", type=int, default=0)
parser.add_argument("--gen-with-mbart", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=96)

parser.add_argument("--model-path", type=str, default=None)

args = parser.parse_args()

# Fill your mbart or bart checkpoint path
if args.gen_with_mbart:
    bart = BARTModel.from_pretrained(
        args.model_path,
        checkpoint_file='model.pt',
        data_name_or_path=args.model_path,
        bpe="sentencepiece",
        layernorm_embedding=True,
    )
else:
    bart = BARTModel.from_pretrained(
        os.path.join('/checkpoint/chuntinz/container/bart.large'),
        checkpoint_file='model.pt',
        data_name_or_path=os.path.join('/checkpoint/chuntinz/container/gpt2_bpe')
    )

noise_params = dict()
noise_params['mask_length'] = args.mask_length
noise_params['mask_ratio'] = args.mask_ratio
noise_params['random_ratio'] = args.random_ratio
noise_params['poisson_lambda'] = args.poisson_lambda
noise_params['replace_length'] = args.replace_length
noise_params['insert_ratio'] = args.insert_ratio
noise_params['mask_whole_word'] = args.mask_whole_word

noise_params['low_mask_prob'] = args.low_mask_prob
noise_params['high_mask_prob'] = args.high_mask_prob
noise_params['low_random_prob'] = args.low_random_prob
noise_params['high_random_prob'] = args.high_random_prob
noise_params['random_word_span'] = args.random_word_span
noise_params['seed'] = 15213

src_input = args.input
bsz = args.batch_size
bart.load_noise_hyperparam(noise_params)

bart.cuda()
bart.eval()
bart.half()
count = 1

fout = open(args.output, 'w', encoding="utf-8")
for _ in range(args.iters):
    with open(src_input, "r", encoding='utf-8') as source:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch, noised_src = bart.sample(slines, beam=4, lenpen=1.0, max_len_a=2.0, max_len_b=0,
                                                               no_repeat_ngram_size=3)

                for src, hypothesis, noise in zip(slines, hypotheses_batch, noised_src):
                    fout.write("S-\t{}\n".format(src))
                    fout.write("NS-\t{}\n".format(noise))
                    fout.write("H-\t{}\n".format(hypothesis))
                    fout.write("\n")
                    fout.flush()
                slines = []

            if count % 500 == 0:
                print("processed {} batches!".format(count))
            slines.append(sline.strip())
            count += 1

        if slines != []:
            hypotheses_batch, noised_src = bart.sample(slines, beam=4, lenpen=3.0, max_len_a=2.0, max_len_b=0,
                                                       no_repeat_ngram_size=3)
            for src, hypothesis, noise in zip(slines, hypotheses_batch, noised_src):
                fout.write("S-\t{}\n".format(src))
                fout.write("NS-\t{}\n".format(noise))
                fout.write("H-\t{}\n".format(hypothesis))
                fout.write("\n")
                fout.flush()
fout.close()