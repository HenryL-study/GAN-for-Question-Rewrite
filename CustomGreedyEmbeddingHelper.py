from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.helper import GreedyEmbeddingHelper


class CustomGreedyEmbeddingHelper(GreedyEmbeddingHelper):
    def __init__(self, embedding, start_tokens, end_token, cnn_context):
        """Initializer.
        Args:
        embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`. The returned tensor
            will be passed to the decoder input.
        start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
        end_token: `int32` scalar, the token that marks end of decoding.
        cnn_context: [batch_size, emb_dim]
        Raises:
        ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
            scalar.
        """
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = tf.concat([self._embedding_fn(self._start_tokens), cnn_context], 1) #[batch_size, emb_dim*2]


    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: tf.concat([self._embedding_fn(sample_ids), cnn_context], 1) #[batch_size, emb_dim*2] 
        return (finished, next_inputs, state)