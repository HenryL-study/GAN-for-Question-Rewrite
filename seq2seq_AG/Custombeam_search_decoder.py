from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import BeamSearchDecoder
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

class CustomBeamSearchDecoder(BeamSearchDecoder):
    def __init__(self,
               cell,
               embedding,
               start_tokens,
               end_token,
               initial_state,
               beam_width,
               output_layer=None,
               cnn_context,
               length_penalty_weight=0.0):
        """Initialize BeamSearchDecoder.
        Args:
        cell: An `RNNCell` instance.
        embedding: A callable that takes a vector tensor of `ids` (argmax ids),
            or the `params` argument for `embedding_lookup`.
        start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
        end_token: `int32` scalar, the token that marks end of decoding.
        initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        beam_width:  Python integer, the number of beams.
        output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
            to storing the result or sampling.
        length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
        Raises:
        TypeError: if `cell` is not an instance of `RNNCell`,
            or `output_layer` is not an instance of `tf.layers.Layer`.
        ValueError: If `start_tokens` is not a vector or
            `end_token` is not a scalar.
        """
        if not rnn_cell_impl._like_rnncell(cell):  # pylint: disable=protected-access
            raise TypeError("cell must be an RNNCell, received: %s" % type(cell))
        if (output_layer is not None
            and not isinstance(output_layer, layers_base.Layer)):
            raise TypeError(
            "output_layer must be a Layer, received: %s" % type(output_layer))
        self._cell = cell
        self._output_layer = output_layer

        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
            lambda ids: embedding_ops.embedding_lookup(embedding, ids))

        self._start_tokens = ops.convert_to_tensor(
            start_tokens, dtype=dtypes.int32, name="start_tokens")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._end_token = ops.convert_to_tensor(
            end_token, dtype=dtypes.int32, name="end_token")
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")

        self._batch_size = array_ops.size(start_tokens)
        self._beam_width = beam_width
        self._length_penalty_weight = length_penalty_weight
        self._initial_cell_state = nest.map_structure(
            self._maybe_split_batch_beams,
            initial_state, self._cell.state_size)
        self._start_tokens = array_ops.tile(
            array_ops.expand_dims(self._start_tokens, 1), [1, self._beam_width]) #[batch_size, beam_width]
        self._start_inputs = self._embedding_fn(self._start_tokens)       #[batch_size, emb_dim, beam_width]
        self.cnn_inputs = array_ops.tile(array_ops.expand_dims(cnn_context, 2), [1, 1, self._beam_width])
        self._start_inputs = tf.concat([self._start_inputs, self.cnn_inputs], 1) #[batch_size, emb_dim*2, beam_width]
        print("beam inputs: ",self._start_inputs)
        self._finished = array_ops.zeros(
            [self._batch_size, self._beam_width], dtype=dtypes.bool)
      
    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
        Args:
        time: scalar `int32` tensor.
        inputs: A (structure of) input tensors.
        state: A (structure of) state tensors and TensorArrays.
        name: Name scope for any created operations.
        Returns:
        `(outputs, next_state, next_inputs, finished)`.
        """
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token
        length_penalty_weight = self._length_penalty_weight

        with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
            cell_state = state.cell_state
            inputs = nest.map_structure(
                lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
            cell_state = nest.map_structure(
                self._maybe_merge_batch_beams,
                cell_state, self._cell.state_size)
            cell_outputs, next_cell_state = self._cell(inputs, cell_state)
            cell_outputs = nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
            next_cell_state = nest.map_structure(
                self._maybe_split_batch_beams,
                next_cell_state, self._cell.state_size)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                end_token=end_token,
                length_penalty_weight=length_penalty_weight)

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = control_flow_ops.cond(
                math_ops.reduce_all(finished), lambda: self._start_inputs,
                lambda: tf.concat([self._embedding_fn(sample_ids), self.cnn_inputs], 1))

        return (beam_search_output, beam_search_state, next_inputs, finished)