import sys
import os

iters = int(sys.argv[1])
root_dir = sys.argv[2]
opt = sys.argv[3]
prefix = sys.argv[4]

if len(sys.argv) > 5:
    suffix = sys.argv[5]
else:
    suffix = "spm"

all_files = []
for fname in os.listdir(root_dir):
    if fname.startswith(prefix) and fname.endswith(suffix):
        all_files.append(fname)

sorted_files = sorted(all_files)

def read_data(fname):
    data = []
    with open(fname, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(line)
    return data

with open(opt, 'w', encoding='utf-8') as fout:
    for fname in sorted_files:
        data = read_data(os.path.join(root_dir, fname))
        for _ in range(iters):
            for line in data:
                fout.write(line)