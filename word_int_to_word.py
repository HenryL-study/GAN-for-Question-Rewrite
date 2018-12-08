import codecs

index_to_word = 'data/Computer/index_to_word.txt'
sample_file = 'save/generator_sample.txt10'
out_file = 'generator_sentence10.txt'

int_to_word={0: "<pad>"}
in_w = codecs.open(index_to_word,'r', 'utf-8')
for w in in_w.readlines():
    wid, word = w.strip().split(" ")
    int_to_word[int(wid)] = word
in_w.close()

sample = codecs.open(sample_file, 'r', 'utf-8')
out = codecs.open(out_file, 'w', 'utf-8')

i = 0
for line in sample.readlines():
    idx = line.strip().split()
    strs = ""
    for id in idx:
        strs = strs + int_to_word[int(id)] + " "
    out.write(strs + '\n')
    i += 1
