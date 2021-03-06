#-*- coding:utf-8 -*-
from __future__ import print_function
import tensorflow as tf
# from Conv_lstm_cell import ConvLSTMCell
from tensorflow.python.layers.core import Dense
from CustomGreedyEmbeddingHelper import CustomGreedyEmbeddingHelper
from Custombeam_search_decoder import CustomBeamSearchDecoder

class Seq2seq_Model(object):

    def __init__(self, num_emb, batch_size, emb_dim, emb_data,
                 ques_length, ans_length, start_token, end_token, gen_filter_sizes, gen_num_filters,
                 isTrain = False, usepre_emb = False, learning_rate=0.0005):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.emb_data = emb_data
        self.max_ques_length = ques_length
        self.max_ans_length = ans_length
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.gen_filter_sizes = gen_filter_sizes
        self.gen_num_filters = gen_num_filters
        self.isTrain = isTrain
        self.usepre_emb = usepre_emb
        self.grad_clip = 5.0

        self.seq_start_token = start_token
        self.seq_end_token = end_token
        self.rnn_size = 1024
        self.layer_size = 2
        self.beam_width = 10
        self.atten_depth = 512 #The depth of the query mechanism

        self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
        
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_ques_length]) # sequence of tokens generated by generator
        self.response = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_ans_length]) # get from rollout policy and discriminator
        self.target_sequence_length = tf.placeholder(tf.int32, [self.batch_size], name='target_sequence_length')
        self.target_response_length = tf.placeholder(tf.int32, [self.batch_size], name='target_response_length')
        self.max_response_length_per_batch = tf.placeholder(tf.int32, shape=())

        with tf.device("/cpu:0"):
            #self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim
            self.processed_x = tf.nn.embedding_lookup(self.g_embeddings, self.x)
            self.processed_response = tf.nn.embedding_lookup(self.g_embeddings, self.response)
            print("processed_x shape: ", self.processed_x.shape)
            print("processed_response shape: ", self.processed_response.shape)

        self.add_encoder_layer()
        self.getCnnEncoder(self.gen_filter_sizes, self.gen_num_filters)
        self.output_layer = Dense(self.num_emb, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
        if self.isTrain:
            decoder_output = self.add_decoder_for_training()
            self.g_pretrain_predictions = decoder_output.rnn_output
            self.g_samples = decoder_output.sample_id

            masks = tf.sequence_mask(self.target_sequence_length, self.max_response_length_per_batch, dtype=tf.float32, name='masks')
            self.train_loss = tf.contrib.seq2seq.sequence_loss(
                self.g_pretrain_predictions,
                self.response[:,0:self.max_response_length_per_batch],
                masks)
            train_opt = self.g_optimizer(self.learning_rate)
            gradients = train_opt.compute_gradients(self.train_loss)
            self.pretrain_grad_zip = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.pretrain_updates = train_opt.apply_gradients(self.pretrain_grad_zip)

        else:
            decoder_output = self.add_decoder_for_inference()
            self.g_samples = decoder_output.predicted_ids


    def init_matrix(self, shape):
        if self.usepre_emb:
            embeddings = tf.get_variable("embeddings", shape=self.emb_data.shape, initializer=tf.constant_initializer(self.emb_data), trainable=True)
        else:
            embeddings = tf.truncated_normal(shape, stddev=0.01)
        return embeddings

    def lstm_cell(self, rnn_size=None):
        rnn_size = self.rnn_size if rnn_size is None else rnn_size
        return tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.orthogonal_initializer())

    def add_encoder_layer(self):
        self.encoder_out = self.processed_x
        for n in range(self.layer_size):
            (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = self.lstm_cell(self.rnn_size // 2),
                cell_bw = self.lstm_cell(self.rnn_size // 2),
                inputs = self.encoder_out,
                sequence_length = self.target_sequence_length,
                dtype = tf.float32,
                scope = 'bidirectional_rnn_'+str(n))
            self.encoder_out = tf.concat((out_fw, out_bw), 2)
        
        bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
        bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
        bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
        self.encoder_state = tuple([bi_lstm_state] * self.layer_size)
        print("encoder state: ", self.encoder_state)
        print("encoder output ", self.encoder_out)

    def add_attention(self):
        if self.isTrain:
            memory = self.encoder_out
            encoder_final_state = self.encoder_state
            memory_sequence_length = self.target_sequence_length
        else:      
            memory = tf.contrib.seq2seq.tile_batch(self.encoder_out, self.beam_width)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, self.beam_width)
            memory_sequence_length = tf.contrib.seq2seq.tile_batch(self.target_sequence_length, self.beam_width)
        
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units = self.rnn_size, 
            memory = memory,
            memory_sequence_length = memory_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_size)]),
            attention_mechanism = attention_mechanism,
            attention_layer_size = self.atten_depth)
        decoder_initial_state = decoder_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=encoder_final_state)
        
        return decoder_cell, decoder_initial_state
        


    def add_decoder_for_training(self):
        cell, initial_state = self.add_attention()
        train_context = tf.expand_dims(self.cnn_context, 1)
        train_seq_inputs = tf.concat([self.processed_response, tf.tile(train_context, [1,self.max_ans_length,1])], 2)
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=train_seq_inputs,
                                                            sequence_length=self.target_response_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                            training_helper,
                                                            initial_state,
                                                            self.output_layer) 
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                    impute_finished=True,
                                                                    maximum_iterations=self.max_ans_length)
        print("training decoder output: ", training_decoder_output)
        return training_decoder_output

    def add_decoder_for_inference(self):
        cell, initial_state = self.add_attention()
        start_tokens = tf.tile(tf.constant([self.seq_start_token], dtype=tf.int32), [self.batch_size], 
                                name='start_tokens')
        predicting_decoder = CustomBeamSearchDecoder(cell=cell,
                                                embedding=self.g_embeddings,
                                                start_tokens=start_tokens,
                                                end_token=self.seq_end_token,
                                                initial_state = initial_state,
                                                beam_width=self.beam_width,
                                                cnn_context = self.cnn_context,
                                                output_layer=self.output_layer,
                                                length_penalty_weight=0.0)

        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                            impute_finished=False,
                                                            maximum_iterations=self.max_ans_length)
        print("predict decoder output: ", predicting_decoder_output)
        return predicting_decoder_output

    def generate(self, sess, x, x_len, response, res_len):
        res_len_max = max(res_len)
        outputs = sess.run(self.g_samples, feed_dict={self.x: x, self.target_sequence_length: x_len, self.response: response, self.target_response_length: res_len, self.max_response_length_per_batch: res_len_max})
        if not self.isTrain:
            outputs = outputs[:,:,0]
        return outputs

    def train_step(self, sess, x, x_len, response, res_len):
        if not self.isTrain:
            print("Predicting model. Can not train.")
            return None
        res_len_max = max(res_len)
        outputs = sess.run([self.train_loss, self.pretrain_updates], feed_dict={self.x: x, self.target_sequence_length: x_len, self.response: response, self.target_response_length: res_len, self.max_response_length_per_batch: res_len_max})
        return outputs
    
    def train_test_step(self, sess, x, x_len, response, res_len):
        res_len_max = max(res_len)
        outputs = sess.run(self.train_loss, feed_dict={self.x: x, self.target_sequence_length: x_len, self.response: response, self.target_response_length: res_len, self.max_response_length_per_batch: res_len_max})
        return outputs

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)

    #define cnn network function
    # An alternative to tf.nn.rnn_cell._linear function, which has been removed in Tensorfow 1.0.1
    # The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
    def linear(self, input_, output_size, scope=None):
        '''
        Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
        Args:
        input_: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    '''

        shape = input_.get_shape().as_list()
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
        input_size = shape[1]

        # Now the computation.
        with tf.variable_scope(scope or "SimpleLinear"):
            matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
            bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

        return tf.matmul(input_, tf.transpose(matrix)) + bias_term

    def highway(self, input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """

        with tf.variable_scope(scope):
            for idx in range(num_layers):
                g = f(self.linear(input_, size, scope='highway_lin_%d' % idx))

                t = tf.sigmoid(self.linear(input_, size, scope='highway_gate_%d' % idx) + bias)

                output = t * g + (1. - t) * input_
                input_ = output

        return output
    
    def getCnnEncoder(self, filter_sizes, num_filters, l2_reg_lambda=0.2):
        self.embedded_chars_expanded = tf.expand_dims(self.processed_x, -1)
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.emb_dim, 1, num_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.max_ques_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = sum(num_filters)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add highway
        with tf.name_scope("highway"):
            self.h_highway = self.highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_highway, 0.75)
        
        with tf.name_scope("cnncontext"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, self.emb_dim], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.emb_dim]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            cnn_context = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        
        self.cnn_context = cnn_context #[batch_size, emb_dim]



    
