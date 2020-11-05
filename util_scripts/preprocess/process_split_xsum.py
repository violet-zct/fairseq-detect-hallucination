import os
import json

root = "/private/home/chuntinz/work/data/summarization"
root_input_dir = "/private/home/chuntinz/work/data/summarization/bbc-summary-data"


def load_json(fname):
    dictionary = json.loads(open(fname).read())
    master = {'test': set(), 'train': set(), 'validation': set()}
    for key in master.keys():
        master[key] = set(dictionary[key])
    return master


splits = ['test', 'train', 'validation']
split_master = load_json(os.path.join(root, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json'))
data = {'test': [], 'train': [], 'validation': []}

for fname in os.listdir(root_input_dir):
    if not fname.endswith("summary"):
        continue
    docid = fname.split(".")[0]
    fin = open(os.path.join(root_input_dir, fname), 'r', encoding='utf-8')
    content = fin.read()
    fin.close()
    fields = content.strip().split('\n\n')
    summary = fields[2].split('[SN]FIRST-SENTENCE[SN]')[-1].strip()
    doc = fields[3].split('[SN]RESTBODY[SN]')[-1].strip()
    for key in splits:
        if docid in split_master[key]:
            data[key].append((doc, summary, docid))

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = set(['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"])# acceptable ways to end a sentence


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + "."


for key in splits:
    if key == 'validation':
        kk = "valid"
    else:
        kk = key

    print(key, len(data[key]))
    if key == 'test':
        fid = open(os.path.join(root, "raw", "test.docid"), "w", encoding='utf-8')
    else:
        fid = None
    with open(os.path.join(root, "raw", "{}.document".format(kk)), "w", encoding='utf-8') as fout1, \
            open(os.path.join(root, "raw", "{}.summary".format(kk)), "w", encoding='utf-8') as fout2:
        for dd, ss, idx in data[key]:
            lines = [fix_missing_period(line).strip() for line in dd.split('\n')]
            # Make article into a single string
            article = ' '.join(lines)
            fout1.write(article.strip() + '\n')
            fout2.write(ss + '\n')

            if fid is not None:
                fid.write(idx + '\n')
    if fid is not None:
        fid.close()
