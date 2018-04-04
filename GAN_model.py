# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import codecs
from dataloader import Gen_Data_loader, Dis_dataloader
from generator_my import Generator
from discriminator import Discriminator

#TODO NO USE
#from rollout import ROLLOUT
#from target_lstm import TARGET_LSTM
#import _pickle as cPickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 200 # sequence length TODO need processing data
START_TOKEN = 1 #
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
BATCH_SIZE = 16
gen_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
gen_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 32
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 16 #64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 2500 #TODO
SEED = 88

generated_num = 280000 #10000

sess = tf.InteractiveSession()

#Parameters
src_vocab_size = None
embedding_size = None
glove_embedding_filename = 'glove-vec.npy'

#Word embedding
def loadGloVe(filename):
    # vocab = []
    # embd = []
    # vocab.append('unk') #装载不认识的词
    # embd.append([0]*embedding_size) #这个emb_size可能需要指定
    # file = codecs.open(filename, 'r', 'utf-8')
    # for line in file.readlines():
    #     row = line.strip().split(' ')
    #     vocab.append(row[0])
    #     embd.append(row[1:])
    # print('GloVe loaded.')
    # file.close()
    embd = np.load(filename)
    return embd

def generate_samples(sess, trainable_model, batch_size, generated_num, output_file, data_loader):
    # Generate Samples
    generated_samples = []
    data_loader.reset_pointer()

    for _ in range(int(generated_num / batch_size)):
        batch, ques_len = data_loader.next_batch()
        generated_samples.extend(trainable_model.generate(sess, batch, ques_len))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            if len(poem) != SEQ_LENGTH:
                print("Error: generate len: ", len(poem))
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch, ques_len = data_loader.next_batch()
        g_loss, _, sample = trainable_model.pretrain_step(sess, batch, ques_len)
        # print("sample shape: ", sample[0])
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses), sample

def process_real_data():
    #processing in process_questions.py
    return 'question-vec.txt', 'not use now'  

def get_reward(sess, input_x, rollout_num, generator, discriminator):
    rewards = []
    for i in range(rollout_num):
        for given_num in range(1, SEQ_LENGTH):
            samples = generator.get_samples(sess, input_x, given_num, [SEQ_LENGTH for _ in range(BATCH_SIZE)])
            feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[given_num - 1] += ypred

        # the last token reward
        feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
        ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
        ypred = np.array([item[1] for item in ypred_for_auc])
        if i == 0:
            rewards.append(ypred)
        else:
            rewards[SEQ_LENGTH-1] += ypred

    rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
    return rewards

random.seed(SEED)
np.random.seed(SEED)

#Word embedding parameters
embedding = loadGloVe(glove_embedding_filename)
embedding_size = embedding.shape[1]
src_vocab_size = embedding.shape[0]
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
# embedding = np.asarray(embd)
#vocab to int
# vocab_to_int = {}
# for i in range(src_vocab_size):
#     vocab_to_int[vocab[i]] = i

print('Glove vector loaded. Total vocab: ', src_vocab_size, '. embedding_size: ', embedding_size)

gen_data_loader = Gen_Data_loader(BATCH_SIZE)
#likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing

dis_data_loader = Dis_dataloader(BATCH_SIZE)

generator = Generator(src_vocab_size, BATCH_SIZE, embedding_size, HIDDEN_DIM, embedding, SEQ_LENGTH, START_TOKEN, gen_filter_sizes, gen_num_filters)
# target_params = cPickle.load(open('save/target_params_py3.pkl', 'rb'))
# target_lstm = TARGET_LSTM(src_vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

#TODO change discriminator's embedding layer
discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=src_vocab_size, embedding_size=embedding_size, 
                            filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False #True
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# First, creat the positive examples
#generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
positive_file, eval_file = process_real_data()
negative_file = 'save/generator_sample.txt'
positive_len_file = 'question-len.txt'
gen_data_loader.create_batches(positive_file, positive_len_file)

log = open('save/experiment-log.txt', 'w')
'''
Loading model. To retrain model uncomment to line 235
'''
#  pre-train generator
print ('Start pre-training...')
log.write('pre-training...\n')
sample = None
for epoch in range(PRE_EPOCH_NUM):
    loss, sample = pre_train_epoch(sess, generator, gen_data_loader)
    if epoch % 5 == 0:
        # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file, gen_data_loader)
        # likelihood_data_loader.create_batches(eval_file)
        # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        print ('pre-train epoch ', epoch, 'generator_loss ', loss)
        buffer = 'epoch:\t'+ str(epoch) + '\tgenerator_loss:\t' + str(loss) + '\n'
        log.write(buffer)

print ('sample: ', sample)
print ('Start pre-training discriminator...')
# Train 3 epoch on the generated data and do this for 50 times
for _ in range(50):
    generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, gen_data_loader)
    dis_data_loader.load_train_data(positive_file, positive_len_file, negative_file)
    for _ in range(3):
        dis_data_loader.reset_pointer()
        for it in range(dis_data_loader.num_batch):
            x_batch, y_batch, x_len = dis_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: dis_dropout_keep_prob
            }
            _, d_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
        
    print("d_loss: ", d_loss)

#test discrimanator
# dis_data_loader.load_train_data(positive_file, positive_len_file, negative_file)
# dis_data_loader.reset_pointer()
# for it in range(dis_data_loader.num_batch):
#     x_batch, y_batch, x_len = dis_data_loader.next_batch()
#     feed = {
#         discriminator.input_x: x_batch,
#         discriminator.input_y: y_batch,
#         discriminator.dropout_keep_prob: dis_dropout_keep_prob
#     }
#     _, d_predict = sess.run([discriminator.train_op, discriminator.predictions], feed)
#     print("x_batch: ", x_batch)
#     print("y_batch: ", y_batch)
#     print("d_predict: ", d_predict)
#     break


# merge rollout into genrater so the update rate 0.2->1(real time). Any side effects?
# rollout = ROLLOUT(generator, 0.8)
print("saving model...")
saver = tf.train.Saver()
saver.save(sess, "save/model/model.ckpt")
'''
Loading model part.
'''
# print("loading model...")
# saver = tf.train.Saver()
# saver.restore(sess, "save/model/model.ckpt")


'''
TEST BEGIN @3.29
'''
 
print ('#########################################################################')
print ('Start Adversarial Training...')
log.write('adversarial training...\n')
sampel_log = open('save/sample-log.txt', 'w')
gen_data_loader.reset_pointer()
for total_batch in range(TOTAL_BATCH):
    # Train the generator for one step
    samples = None
    for it in range(1):
        batch, ques_len = gen_data_loader.next_batch()
        samples = generator.generate(sess, batch, ques_len)
        rewards = get_reward(sess, samples, 16, generator, discriminator)
        # print("rewards sample: ", rewards[0])
        feed = {generator.x: samples, generator.rewards: rewards, generator.target_sequence_length: ques_len, generator.max_sequence_length_per_batch: max(ques_len)}
        _, g_loss = sess.run([generator.g_updates, generator.g_loss], feed_dict=feed)
    
    buffer = 'epoch:\t' + str(total_batch) + '\tg_loss:\t' + str(g_loss) + '\n'
    print ('total_batch: ', total_batch, 'g_loss: ', g_loss)
    log.write(buffer)
    # write sample
    if total_batch % 10 == 0 or total_batch == TOTAL_BATCH - 1:
        # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file, gen_data_loader)
        # likelihood_data_loader.create_batches(eval_file)
        # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
        for seq in samples:
            for word in seq:
                sampel_log.write(str(word))
                sampel_log.write(" ")
            sampel_log.write("\n")
        
        print("rewards sample: ", rewards[0])



    # Update roll-out parameters
    #rollout.update_params()

    # Train the discriminator
    for _ in range(5):
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file, gen_data_loader)
        dis_data_loader.load_train_data(positive_file, positive_len_file, negative_file)

        d_loss = 0
        for _ in range(3):
            dis_data_loader.reset_pointer()
            d_t = 0
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch, x_len = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob
                }
                _, d_temp = sess.run([discriminator.train_op, discriminator.loss], feed)
                d_t += d_temp
            d_loss += d_t
        d_loss = d_loss/3
        print ('\t\t\t\t\ttotal_batch: ', total_batch, 'd_loss: ', d_loss)

    print("saving model...")
    saver = tf.train.Saver()
    saver.save(sess, "save/model/model.ckpt")

log.close()
sampel_log.close()





