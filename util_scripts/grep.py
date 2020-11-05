import sys

input = sys.argv[1]
fhypo = sys.argv[2]
fref = sys.argv[3]
total = 0
H = S = NS = 0
with open(input, 'r', encoding='utf-8') as fin, open(fhypo, 'w', encoding='utf-8') as fh, open(fref, 'w', encoding='utf-8') as fr:
    for line in fin:
        if line.strip() == '':
            total += 1
        if line.startswith("H-"):
            H += 1
            fh.write(line.strip().split('\t')[-1] + '\n')
        if line.startswith('NS-'):
            NS += 1
        if line.startswith('S-'):
            S += 1
            fr.write(line.strip().split('\t')[-1] + '\n')

print('total={}, H={}, S={}, NS={}'.format(total, H, S, NS))
