import codecs

index_to_word = 'data/Informatique/index_to_word.txt'
sample_file = 'save/sample-log.txt'

int_to_word={0: "<pad>"}
in_w = codecs.open(index_to_word,'r', 'utf-8')
for w in in_w.readlines():
    wid, word = w.strip().split(" ")
    int_to_word[wid] = word
in_w.close()

sample = codecs.open(sample_file, 'r', 'utf-8')

for line in sample.readlines():
    idx = line.strip().split()
    str = ""
    for id in idx:
        str = str + int_to_word[int(id)] + " "
    print(str)

