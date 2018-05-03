#-*- coding:utf-8 -*-
from __future__ import print_function
import os
import codecs
import re
import tensorflow
import numpy as np



from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#Parameters
embedding_size = 200
glove_embedding_filename = 'data/glove.twitter.27B.200d.txt'
question_filename = 'data/Computer/Computers&Internet.txt' #'question-simple.txt'
question_gen_filename = 'data/Computer/generator_sentence17.txt' #'question-simple.txt'
ans_filename = 'data/Computer/Computers&Internet_ans.txt' #'question-simple.txt'

processed_filename = 'data/Computer/question-vec.txt'
processed_ques_len = 'data/Computer/question-len.txt'
processed_ansname = 'data/Computer/answer-vec.txt'
processed_ans_len = 'data/Computer/answer-len.txt'
processed_catname = 'data/Computer/concat-vec.txt'
processed_cat_len = 'data/Computer/concat-len.txt'
processed_glove = 'data/Computer/glove-vec'
index_to_word = 'data/Computer/index_to_word.txt'


ques = []
ques_concat = []
MAX_LENGTH = 0
file = open(question_filename,'r')
for line in file.readlines():
    row = 'starttrats ' + line.strip() + ' enddne'
    row_c = 'starttrats ' + line.strip()
    ques.append(row)
    ques_concat.append(row_c)
    row_ = text_to_word_sequence(row)
    MAX_LENGTH = max(MAX_LENGTH, len(row_))
file.close()

file = open(question_gen_filename,'r')
i = 0
for line in file.readlines():
    if line.strip() == '':
        i-=1
        break
    row = 'starttrats ' + line.strip() + ' enddne'
    row_ = text_to_word_sequence(row)
    MAX_LENGTH = max(MAX_LENGTH, len(row_))
    ques_concat[i] = ques_concat[i] + ' ' + row
    i+=1
file.close()
ans = []
ANS_MAX_LENGTH = 0
file = open(ans_filename,'r')
for line in file.readlines():
    row = 'starttrats ' + line.strip().split('.')[0] + ' enddne'
    row_ = text_to_word_sequence(row)
    ANS_MAX_LENGTH = max(ANS_MAX_LENGTH, len(row_))
    ans.append(row)
file.close()

ques = ques[:i]
ques_concat = ques_concat[:i]
ans = ans[:i]

embedding_index = {}
fopen = codecs.open(glove_embedding_filename, 'r', 'utf-8')
i=0
for eachLine in fopen.readlines():
    # First element in each line is the word
    values = eachLine.split()
    if len(values) < 2:
        print(i)
    word = values[0]
    # Word vectors
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
    i+=1
fopen.close()
embedding_index['starttrats'] = np.asarray(['0' for _ in range(embedding_size)], dtype='float32')
embedding_index['enddne'] = np.asarray(['0' for _ in range(embedding_size)], dtype='float32')

print('Found %s word vectors.' % len(embedding_index))

total_texts = []
total_texts = ques + ans + ques_concat
tokenizer = Tokenizer()
tokenizer.fit_on_texts(total_texts)

sequences_train = tokenizer.texts_to_sequences(ques)
ans_train = tokenizer.texts_to_sequences(ans)
concat_train = tokenizer.texts_to_sequences(ques_concat)

ques_len = codecs.open(processed_ques_len,'w', 'utf-8')
ques_len_static = [0,0,0,0,0,0,0]
for seq in sequences_train:
    if len(seq) < 50: 
        ques_len_static[0] += 1
        ques_len.write(str(len(seq)))
        ques_len.write(" ")
    elif len(seq) < 100:
        ques_len_static[1] += 1
        ques_len.write(str(len(seq)))
        ques_len.write(" ")
    elif len(seq) < 200:
        ques_len_static[2] += 1
        ques_len.write("100")
        ques_len.write(" ")
    elif len(seq) < 300:
        ques_len_static[3] += 1
        ques_len.write("100")
        ques_len.write(" ")
    elif len(seq) < 400:
        ques_len_static[4] += 1
        ques_len.write("100")
        ques_len.write(" ")
    elif len(seq) < 500:
        ques_len_static[5] += 1
        ques_len.write("100")
        ques_len.write(" ")
    else:
        ques_len_static[6] += 1
        ques_len.write("100")
        ques_len.write(" ")
ques_len.close()
print("ques_len_static:\n", ques_len_static)

ans_len = codecs.open(processed_ans_len,'w', 'utf-8')
ans_len_static = [0,0,0,0,0,0,0]
for seq in ans_train:
    if len(seq) < 5: 
        ans_len_static[0] += 1
        ans_len.write(str(len(seq)))
        ans_len.write(" ")
    elif len(seq) < 10:
        ans_len_static[1] += 1
        ans_len.write(str(len(seq)))
        ans_len.write(" ")
    elif len(seq) < 15:
        ans_len_static[2] += 1
        ans_len.write(str(len(seq)))
        ans_len.write(" ")
    elif len(seq) < 20:
        ans_len_static[3] += 1
        ans_len.write(str(len(seq)))
        ans_len.write(" ")
    elif len(seq) < 25:
        ans_len_static[4] += 1
        ans_len.write("20")
        ans_len.write(" ")
    elif len(seq) < 30:
        ans_len_static[5] += 1
        ans_len.write("20")
        ans_len.write(" ")
    else:
        ans_len_static[6] += 1
        ans_len.write("20")
        ans_len.write(" ")
ans_len.close()
print("ans_len_static:\n", ans_len_static)

cat_len = codecs.open(processed_cat_len,'w', 'utf-8')
cat_len_static = [0,0,0,0,0,0,0]
for seq in concat_train:
    if len(seq) < 50: 
        cat_len_static[0] += 1
        cat_len.write(str(len(seq)))
        cat_len.write(" ")
    elif len(seq) < 100:
        cat_len_static[1] += 1
        cat_len.write(str(len(seq)))
        cat_len.write(" ")
    elif len(seq) < 150:
        cat_len_static[2] += 1
        cat_len.write(str(len(seq)))
        cat_len.write(" ")
    elif len(seq) < 200:
        cat_len_static[3] += 1
        cat_len.write(str(len(seq)))
        cat_len.write(" ")
    elif len(seq) < 300:
        cat_len_static[4] += 1
        cat_len.write("200")
        cat_len.write(" ")
    elif len(seq) < 400:
        cat_len_static[5] += 1
        cat_len.write("200")
        cat_len.write(" ")
    else:
        cat_len_static[6] += 1
        cat_len.write("200")
        cat_len.write(" ")
cat_len.close()
print("cat_len_static:\n", cat_len_static)

#print(ques[0])
#print(sequences_train[0])
# # Auto filled with 0
# remove MAX_LENGTH setting below to use the max length of all sentences.
MAX_LENGTH = 100
data_train = pad_sequences(sequences_train, maxlen = MAX_LENGTH, padding='post', truncating='post')
ANS_MAX_LENGTH = 20
ans_train = pad_sequences(ans_train, maxlen = ANS_MAX_LENGTH, padding='post', truncating='post')
CAT_MAX_LENGTH = 200
cat_train = pad_sequences(concat_train, maxlen = CAT_MAX_LENGTH, padding='post', truncating='post')


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
print('Start token: ', word_index['starttrats'])
print('End token: ', word_index['enddne'])

for ans in ans_train:
    if ans[ANS_MAX_LENGTH-1] != word_index['enddne'] or ans[ANS_MAX_LENGTH-1] != 0:
        ans[ANS_MAX_LENGTH-1] = word_index['enddne']
# Prepare embedding matrix
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, embedding_size))
in_to_word = {}
for word, i in word_index.items():
    #print(word)
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all zeros
        embedding_matrix[i] = embedding_vector
    in_to_word[i] = word

in_w = codecs.open(index_to_word,'w', 'utf-8')
for i, word in in_to_word.items():
    in_w.write(str(i) + ' ' + unicode(word, 'utf8')+'\n')
in_w.close()

np.save(processed_glove,embedding_matrix)
np.savetxt(processed_filename,data_train, fmt="%d", delimiter=' ')
np.savetxt(processed_ansname, ans_train, fmt="%d", delimiter=' ')
np.savetxt(processed_catname, cat_train, fmt="%d", delimiter=' ')

print("Processing done.")
print("Max length of Q\tA\tQ_C: ", MAX_LENGTH, ANS_MAX_LENGTH, CAT_MAX_LENGTH)
print("Embedding shape: ", embedding_matrix.shape)
print("Data shape: ", data_train.shape, ans_train.shape, cat_train.shape)


# #Word embedding
# def loadGloVe(filename):
#     vocab = []
#     embd = []
#     vocab.append('unk') #装载不认识的词
#     embd.append([0]*embedding_size) #这个emb_size可能需要指定
#     file = codecs.open(filename, 'r', 'utf-8')
#     for line in file.readlines():
#         row = line.strip().split(' ')
#         vocab.append(row[0])
#         embd.append(row[1:])
#     print('GloVe loaded.')
#     file.close()
#     return vocab,embd


# vocab,embd = loadGloVe(glove_embedding_filename)
# embedding_size = len(embd[0])
# #Add start & end & unknown & pad token
# PAD_TOKEN = 0
# vocab.insert(0, '<p_a_d>')
# embd.insert(0, ['0' for _ in range(embedding_size)])
# START_TOKEN = len(vocab)
# vocab.append('<s_t_a_r_t>')
# embd.append(['0' for _ in range(embedding_size)])
# END_TOKEN = len(vocab)
# vocab.append('<e_n_d>')
# embd.append(['0' for _ in range(embedding_size)])
# UKNOWN_TOKEN = len(vocab)
# vocab.append('<u_k_n_o_w_n>')
# embd.append(['0' for _ in range(embedding_size)])
# src_vocab_size = len(vocab)

# #vocab to int
# vocab_to_int = {}
# for i in range(src_vocab_size):
#     vocab_to_int[vocab[i]] = i

# print('Glove vector loaded. Total vocab: ', src_vocab_size, '. embedding_size: ', embedding_size)

# ques = []
# MAX_LENGTH = 0
# file = codecs.open(question_filename,'r', 'utf-8')
# for line in file.readlines():
#     row = line.strip()
#     row = text_to_word_sequence(row)
#     MAX_LENGTH = max(MAX_LENGTH, len(row))
#     ques.append(row)
# file.close()

# #to int & reconstruct embedding
# re-embed = []
# re-vocab = []
# for q in ques:
#     for word in q:
        
# embedding = np.asarray(embd)

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(ques)

# sequences_ques = tokenizer.texts_to_sequences(ques)
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
# sequences_ques = pad_sequences(sequences_ques)
# print('questions shape: ', sequences_ques.shape)

# # Prepare embedding matrix
# num_words = len(word_index) + 4
# embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if i == 0:
#         print('impossible!!!')
#     if word in vocab:
#         embedding_vector = embedding[vocab.index(word)]
#         embedding_matrix[i] = embedding_vector
