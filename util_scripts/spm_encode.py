import sentencepiece as spm
import sys

input = sys.argv[1]
vocab = sys.argv[2]

if len(sys.argv) > 3:
    opt = sys.argv[3]
else:
    opt = input+".bpe"
sp = spm.SentencePieceProcessor()
sp.Load(vocab)

with open(input, "r", encoding="utf-8") as fin, open(opt, "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")