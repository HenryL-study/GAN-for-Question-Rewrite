# GAN-for-Question-Rewrite

A GAN can rewrite questions then use original questions and rewrite questions to train a seq2seq with CNN model for answer generation.

**Note: the most updated code is in brach `cnnplstm`** 

### Prerequisites

All codes are written in Python with [tensorflow](www.tensorflow.org)

```
tensorflow >= 1.2
```
### Get Started

1. run `process_questions.py` to processing questions & answers.
2. run `run.sh` for training GAN. All GAN parameters are in `GAN_model.py`
3. run `seq2seq_AG/model.py` to train seq2seq model

### Data Set

##### Q&A data

[L6 - Yahoo! Answers Comprehensive Questions and Answers version 1.0 (multi part)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l&guccounter=1)

##### Word Embedding

Download pre-trained word vectors in [Glove](https://nlp.stanford.edu/projects/glove/)

Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download)

### Network Structure

#### GAN

```
|---Generator

|	|

|	|---input:

|	|		[EMB_DIM, SEQ_LENGTH, BATCH_SIZE] = [200, ?, 64]

|	|---layer1.1:

|	|		LSTM_FWD: HIDDEN_STATE = 32

|	|		LSTM_BWD: HIDDEN_STATE = 32

|	|		Attention: context vector = [CONTEXT, POST_DIM, BATCH_SIZE] = [200 64, 32]

|	|---layer1.2:

			conv1-8:

			filter_sizes = [1, 2, 3, 8, 9, 10, 15, 20]

			num_filters  = [100, 200, 100, 100, 100, 100, 160, 160]

|	|---layer2:

|	|		LSTM_POST: HIDDEN_STATE = 64

|	|---layer3:

|	|		FC_1: HIDDEN_UNITS = target_vocab_size

|	|		Softmax

|	|		Argmax Sample

|	|---optimizer:

|			tf.train.AdamOptimizer

|---Discriminator

|	|

|	|---input:

|	|	[EMB_DIM, SEQ_LENGTH, BATCH_SIZE] = [300, ?, 32]

|	|---conv1-12:

|	|	filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

|	|	num_filters  = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

|	|---FC：

|	|	HIDDEN_UNITS = 300

|	|	Softmax

|	|---output:

|	|	[BATCH_SIZE, label] = [32, 2]

|	|---optimizer:

|		tf.train.AdamOptimizer
```



#### Seq2Seq

Most same with generator and add [bead search](https://arxiv.org/abs/1703.01619) to get better perfomance.



