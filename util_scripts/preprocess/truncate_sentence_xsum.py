import sys

fn_doc = sys.argv[1]
fn_sum = sys.argv[2]
fn_ref = sys.argv[3]

opt = sys.argv[4]


def truncate(ldoc, lref, lsum, special_tokens=4):
    ndoc = len(ldoc.strip().split())
    nref = len(lref.strip().split())
    nsum = len(lsum.strip().split())

    if special_tokens == 4:
        final = min(512 - nsum - special_tokens, 512 - nref - special_tokens)
    elif special_tokens == 6:
        final = 512 - nsum - nref - special_tokens

    return " ".join(ldoc.strip().split()[:final]), ndoc - final


for sp_token in [6]:
    fn_out = opt

    num_truncated = 0
    avg_truncate_length = 0
    with open(fn_doc, 'r', encoding='utf-8') as fdoc, \
            open(fn_ref, 'r', encoding='utf-8') as fref, \
            open(fn_sum, 'r', encoding='utf-8') as fsum, \
            open(fn_out, 'w', encoding='utf-8') as fout:
        for ldoc, lref, lsum in zip(fdoc, fref, fsum):
            truncat_doc, truncate_num = truncate(ldoc, lref, lsum, sp_token)
            if truncate_num > 0:
                num_truncated += 1
                avg_truncate_length += truncate_num
            fout.write(truncat_doc + '\n')

    # print("Bi-output: " if sp_token == 4 else "Tri-ouput: ")
    print("{} sentences truncated, avg truncate length = {}".format(num_truncated, avg_truncate_length*1.0/num_truncated))