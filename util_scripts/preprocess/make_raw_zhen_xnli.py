import os

path = "/Users/chuntinz/Downloads/XNLI-1.0"


def read_test_tsv(path, opt1, opt2):
    with open(path, "r", encoding="utf-8") as fin, open(opt1, 'w', encoding='utf-8') as fout1, open(opt2, 'w', encoding='utf-8') as fout2:
        for line in fin:
            fields = line.strip().split('\t')

            lang = fields[0]
            sent1, sent2 = fields[-3], fields[-2]
            label = fields[1]
            promptID, pairID = fields[8], fields[9]

            if lang == "zh":
                fout1.write("{}\t{}\t{}\t{}\t{}\n".format(sent1, sent2, label, promptID, pairID))
            if lang == "en":
                fout2.write("{}\t{}\t{}\t{}\t{}\n".format(sent1, sent2, label, promptID, pairID))


def make_new_test_dev(path1, path2, out_dir, prefix):
    sent1_opt = open(os.path.join(out_dir, "{}.sent1".format(prefix)), "w", encoding="utf-8")
    sent2_opt = open(os.path.join(out_dir, "{}.sent2".format(prefix)), "w", encoding="utf-8")
    label_opt = open(os.path.join(out_dir, "{}.label".format(prefix)), "w", encoding="utf-8")

    with open(path1, "r", encoding="utf-8") as fin1, open(path2, "r", encoding="utf-8") as fin2:
        for l1, l2 in zip(fin1, fin2):
            zh_sent_1, zh_sent_2, label1, zh_promptID, zh_pairID = l1.strip().split('\t')
            en_sent_1, en_sent_2, label2, en_promptID, en_pairID = l2.strip().split('\t')
            assert label1 == label2 and zh_promptID == en_promptID and zh_pairID == en_pairID
            sent1_opt.write(zh_sent_1 + "\n")
            sent2_opt.write(en_sent_2 + "\n")
            if label1 == "contradiction":
                label_opt.write("contradictory\n")
            else:
                label_opt.write(label1 + "\n")
    sent1_opt.close()
    sent2_opt.close()
    label_opt.close()


def make_train(path1, path2, out_dir, prefix):
    sent1_opt = open(os.path.join(out_dir, "{}.sent1".format(prefix)), "w", encoding="utf-8")
    sent2_opt = open(os.path.join(out_dir, "{}.sent2".format(prefix)), "w", encoding="utf-8")
    label_opt = open(os.path.join(out_dir, "{}.label".format(prefix)), "w", encoding="utf-8")

    count = 0
    with open(path1, "r", encoding="utf-8") as fin1, open(path2, "r", encoding="utf-8") as fin2:
        for l1, l2 in zip(fin1, fin2):
            zh_sent_1, zh_sent_2, label1 = l1.strip().split("\t")
            en_sent1_1, en_sent_2, label2 = l2.strip().split("\t")

            if count == 0:
                count += 1
                continue
            count += 1
            assert label1 == label2
            sent1_opt.write(zh_sent_1 + "\n")
            sent2_opt.write(en_sent_2 + "\n")
            label_opt.write(label1 + "\n")

    sent1_opt.close()
    sent2_opt.close()
    label_opt.close()

read_test_tsv(os.path.join(path, "xnli.test.tsv"), os.path.join(path, "zh.test.tsv"), os.path.join(path, "en.test.tsv"))
read_test_tsv(os.path.join(path, "xnli.dev.tsv"), os.path.join(path, "zh.dev.tsv"), os.path.join(path, 'en.dev.tsv'))
make_new_test_dev(os.path.join(path, "zh.test.tsv"), os.path.join(path, "en.test.tsv"), os.path.join(path, "zh_en"), "test")
make_new_test_dev(os.path.join(path, "zh.dev.tsv"), os.path.join(path, "en.dev.tsv"), os.path.join(path, "zh_en"), "valid")

train_path = "/Users/chuntinz/Downloads/XNLI-MT-1.0/multinli"
make_train(os.path.join(train_path, "multinli.train.zh.tsv"), os.path.join(train_path, "multinli.train.en.tsv"), os.path.join(path, "zh_en"), "train")