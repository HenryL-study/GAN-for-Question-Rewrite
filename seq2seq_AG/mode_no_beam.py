# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import codecs
from dataloader import Data_loader
from seq2seq_model_no_beam import Seq2seq_Model

#########################################################################################
#  Hyper-parameters
######################################################################################
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 100 # sequence length TODO need processing data
ANS_LENGTH = 20 # sequence length TODO need processing data
START_TOKEN = 1 #
EPOCH_NUM = 500 # supervise (maximum likelihood estimation) epochs
BATCH_SIZE = 64
gen_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
gen_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

SEED = 88

src_vocab_size = None
embedding_size = None
glove_embedding_filename = 'data/Computer/glove-vec.npy'
ques_file = 'data/Computer/concat-vec.txt'
ques_len_file = 'data/Computer/concat-len.txt'
ans_file = 'data/Computer/answer-vec.txt'
ans_len_file = 'data/Computer/answer-len.txt'
gen_ans_file = 'save/gen_cat_ans_sample'


def train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    supervised_g_test_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        ques_batch, ques_len, ans_batch, ans_len = data_loader.next_batch()
        g_loss, _, t_sample, g_sample = trainable_model.train_step(sess, ques_batch, ques_len, ans_batch, ans_len)
        # print("sample shape: ", sample[0])
        if it % 20 == 0:
            print('loss sample in batch ', it, ' : ', g_loss)
        supervised_g_losses.append(g_loss)

    for it in range(data_loader.num_test_batch):
        ques_batch, ques_len, ans_batch, ans_len = data_loader.next_test_batch()
        g_test_loss= trainable_model.train_test_step(sess, ques_batch, ques_len, ans_batch, ans_len)
        # print("sample shape: ", sample[0])
        supervised_g_test_losses.append(g_test_loss)

    return np.mean(supervised_g_losses), np.mean(supervised_g_test_losses), t_sample, g_sample

def get_gen_ans(trainable_model, data_loader, gen_ans_file, epoch):
    generated_answers = []
    data_loader.reset_pointer()
    for _ in range(data_loader.num_test_batch):
        ques_batch, ques_len, ans_batch, ans_len = data_loader.next_test_batch()
        sample, att_img = trainable_model.generate(sess, ques_batch, ques_len, ans_batch, ans_len)
        generated_answers.extend(sample)

    image_file = "image/attention_test" + str(epoch) + ".png"
    print("save attention image to %s*" % image_file)
    image_summ = tf.Summary()
    image_summ.ParseFromString(att_img)
    with tf.gfile.GFile(image_file, mode="w") as img_f:
        img_f.write(image_summ.value[0].image.encoded_image_string)

    with open(gen_ans_file + 'test.txt' + str(epoch), 'w') as fout:
        for poem in generated_answers:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

    generated_answers = []
    data_loader.reset_pointer()
    for _ in range(20):
        ques_batch, ques_len, ans_batch, ans_len = data_loader.next_batch()
        sample, att_img = trainable_model.generate(sess, ques_batch, ques_len, ans_batch, ans_len)
        generated_answers.extend(sample)
        
    image_file = "image/attention_train" + str(epoch) + ".png"
    print("save attention image to %s*" % image_file)
    image_summ = tf.Summary()
    image_summ.ParseFromString(att_img)
    with tf.gfile.GFile(image_file, mode="w") as img_f:
        img_f.write(image_summ.value[0].image.encoded_image_string)

    with open(gen_ans_file + 'train.txt' + str(epoch), 'w') as fout:
        for poem in generated_answers:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

random.seed(SEED)
np.random.seed(SEED)

#Word embedding parameters
embedding = np.load(glove_embedding_filename)
embedding_size = embedding.shape[1]
src_vocab_size = embedding.shape[0]

print('Glove vector loaded. Total vocab: ', src_vocab_size, '. embedding_size: ', embedding_size)

data_loader = Data_loader(BATCH_SIZE, SEQ_LENGTH, ANS_LENGTH)
seq2seq_model = Seq2seq_Model(src_vocab_size, BATCH_SIZE, embedding_size, HIDDEN_DIM, embedding, SEQ_LENGTH, ANS_LENGTH, START_TOKEN, gen_filter_sizes, gen_num_filters)

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9  #allow_growth = False #True
config.allow_soft_placement = True
sess = tf.Session(config = config)
sess.run(tf.global_variables_initializer())

data_loader.create_batches(ques_file, ques_len_file, ans_file, ans_len_file)

# print("loading model...")
# saver = tf.train.Saver()
# saver.restore(sess, "save/model/model.ckpt")
# print ('Done')
#  pre-train generator
print ('Start training...')
for epoch in range(EPOCH_NUM):
    loss, test_loss, sample, g_sample = train_epoch(sess, seq2seq_model, data_loader)
    print ('\t\t\t\ttrain epoch ', epoch, 'train_loss ', loss, 'test_loss ', test_loss)
    if epoch % 10 == 0:
        # print(g_sample)
        get_gen_ans(seq2seq_model, data_loader, gen_ans_file, epoch)

print("saving model...")
saver = tf.train.Saver()
saver.save(sess, "save/model/model_cat.ckpt")
print ('Done')

