Data Set

L6 - Yahoo! Answers Comprehensive Questions and Answers version 1.0 (multi part)    //目前无法下载，无yahoo邮箱

Word Embedding

Download pre-trained word vectors

Pre-trained word vectors.
    Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip

    Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): glove.42B.300d.zip

    Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip
    
    Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip
    
Network Structure
|
|---Generator
|   |
|   |---input:
|   |       [EMB_DIM, SEQ_LENGTH, BATCH_SIZE] = [300, ?, 32]
|   |---layer1:
|   |       LSTM_FWD: HIDDEN_STATE = 32
|   |       LSTM_BWD: HIDDEN_STATE = 32
|   |---layer2:
|   |       Attention: context vector = [CONTEXT, POST_DIM, BATCH_SIZE] = [200 64, 32]
|   |---layer3:
|   |       LSTM_POST: HIDDEN_STATE = 64
|   |---layer4:
|   |       FC_1: HIDDEN_UNITS = 128
|   |       FC_2: HIDDEN_UNITS = 300
|   |       Softmax
|   |       Argmax Sample
|   |---output:
|   |       [EMB_DIM, SEQ_LENGTH, BATCH_SIZE] = [300, ?, 32]
|   |---optimizer:
|           tf.train.AdamOptimizer
|
|
|---Discriminator
|   |
|   |---input:
|   |       [EMB_DIM, SEQ_LENGTH, BATCH_SIZE] = [300, ?, 32]
|   |---conv1-12:
|   |       filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
|   |       num_filters  = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
|   |---FC：
|   |       HIDDEN_UNITS = 300
|   |       Softmax
|   |---output:
|   |       [BATCH_SIZE, label] = [32, 2]
|   |---optimizer:
|           tf.train.AdamOptimizer


Generator

Pre-Training
|
|   loss = -sum(X * log(X_predict)) / N
|
|   pretrain_loss = -tf.reduce_sum(
|               tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
|                   tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
|               )
|           ) / (self.sequence_length * self.batch_size)
|
|   self.x = [self.batch_size, self.sequence_length]
|   self.num_emb = vocab_size
|   self.g_predictions = [batch_size, seq_length, vocab_size]

Unsupervised Training
|
|   loss = -sum(sum(X * log(X_predict)) * rewards)
|
|   self.g_loss = -tf.reduce_sum(
|       tf.reduce_sum(
|           tf.one_hot(tf.to_int32(tf.reshape(self.x, [-1])), self.num_emb, 1.0, 0.0) * tf.log(
|               tf.clip_by_value(tf.reshape(self.g_predictions, [-1, self.num_emb]), 1e-20, 1.0)
|           ), 1) * tf.reshape(self.rewards, [-1])
|   )
