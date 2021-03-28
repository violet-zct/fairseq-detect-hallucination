import sys
import numpy as np
# import nltk
# from nltk.corpus import stopwords

f_reference = sys.argv[1]
f_hypothesis = sys.argv[2]
opt = sys.argv[3]

filter_sw = False
print_backpath = False
output_more = False
# en_stopwords = set(stopwords.words('english'))


def back_trace_edit_distance(str1, str2, print_path=False):
    # minimum edit distance to convert str1 to str2 (str1 is hypo, str2 is reference)
    hypo_len = len(str1)
    ref_len = len(str2)
    dp_cost = np.zeros((hypo_len+1, ref_len+1))
    for ii in range(hypo_len+1):
        dp_cost[ii, 0] = ii
    for ii in range(ref_len+1):
        dp_cost[0, ii] = ii

    for i in range(1, hypo_len+1):
        for j in range(1, ref_len+1):
            dp_cost[i, j] = min(dp_cost[i-1, j] + 1,  # deletion
                                dp_cost[i, j-1] + 1,  # insertion
                                dp_cost[i-1, j-1] + (0 if str1[i-1] == str2[j-1] else 1)  # replacement
                                )
    # back-trace the path
    i = hypo_len
    j = ref_len
    operations = []  # 0: no change; 1: replacement; 2: deletion; 3: insertion
    while i >= 0 or j >= 0:
        if i == 0 and j == 0:
            break
        if i > 0 and dp_cost[i][j] == dp_cost[i-1][j] + 1:
            if print_path: print("position {}: Delete {}".format(i, str1[i - 1]))
            if filter_sw:  # and str1[i-1].strip('\u2581') in en_stopwords:
                operations.append(0)
            else:
                operations.append(2)
            i -= 1
        elif j > 0 and dp_cost[i][j] == dp_cost[i][j-1] + 1:
            if print_path: print("position {}: Insert {}".format(i, str2[j-1]))
            # operations.append(3)
            j -= 1
        elif i > 0 and j > 0 and dp_cost[i][j] == dp_cost[i-1][j-1] + 1:
            if print_path: print("position {}: Replace {} with {}".format(i, str1[i-1], str2[j-1]))
            if filter_sw and str1[i-1].strip('\u2581') in en_stopwords:
                operations.append(0)
            else:
                operations.append(1)
            i -= 1
            j -= 1
        elif i > 0 and j > 0:
            if print_path: print("position {}: unchanged".format(i))
            operations.append(0)
            i -= 1
            j -= 1
        else:
            raise ValueError
    assert i == 0 and j == 0
    return dp_cost[hypo_len, ref_len], operations


fopt = open(opt, "w")
if output_more:
    fopt_detail = open(opt+".full", "w", encoding="utf-8")

with open(f_reference, 'r', encoding='utf-8') as fref, open(f_hypothesis, 'r', encoding='utf-8') as fhypo:
    for ref, hypo in zip(fref, fhypo):
        edit_distance, operations = back_trace_edit_distance(hypo.strip().lower().split(), ref.strip().lower().split(), print_backpath)
        print("ED = {}".format(edit_distance))
        # 1: is hallucination (replace or deletion of words in hypo); 0: not hallucination
        labels = [1 if ii == 1 or ii == 2 else 0 for ii in operations[::-1]]
        assert len(labels) == len(hypo.strip().split())
        fopt.write(" ".join([str(l) for l in labels]) + "\n")

        if output_more:
            fopt_detail.write("R-\t{}\n".format(ref.strip()))
            fopt_detail.write("HL-\t{}\n".format(" ".join("{}[{}]".format(h, l) for h, l in zip(hypo.strip().split(), labels))))
            fopt_detail.write("\n")
fopt.close()

# if __name__ == '__main__':
#     str1 = "saturday"
#     str2 = "sunday"
#
#     str1 = "sunday"
#     str2 = "saturday"
#
#     str1 = "papqd"
#     str2 = "aqdsfzo"
#
#     str2 = "abd"
#     str1 = "jkkkabd"
#     edit_distance, operations = back_trace_edit_distance(str1, str2, print_backpath)
#     print("ED = {}".format(edit_distance))