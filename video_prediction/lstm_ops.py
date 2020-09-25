# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convolutional LSTM implementation."""
# 卷积长短期记忆  实现

import tensorflow as tf

from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers

#初始化状态
def init_state(inputs,
               state_shape,
               state_initializer=tf.zeros_initializer(),
               dtype=tf.float32):
  """Helper function to create an initial state given inputs.
  #创建给定输入的初始状态的辅助函数
  Args:
    inputs: input Tensor, at least 2D, the first dimension being batch_size
    state_shape: the shape of the state.
    state_initializer: Initializer(shape, dtype) for state Tensor.
    dtype: Optional dtype, needed when inputs is None.
    输入张量，至少二维，第一维是batch_size
    状态形状
    状态初始化
    数据类型：可选类型
  Returns:
     A tensors representing the initial state.
     代表初始状态的张量

  """
  if inputs is not None:
    # Handle both the dynamic shape as well as the inferred shape.
    # 处理动态形状 和 推断的形状
    inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
    dtype = inputs.dtype
  else:
    inferred_batch_size = 0
  initial_state = state_initializer(
      [inferred_batch_size] + state_shape, dtype=dtype)
  return initial_state


@add_arg_scope
def basic_conv_lstm_cell(inputs,
                         state,
                         num_channels,
                         filter_size=5,
                         forget_bias=1.0,
                         scope=None,
                         reuse=None):
  """Basic LSTM recurrent network cell, with 2D convolution connctions.
      具有2D卷积连接的基本LSTM递归网络单元
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  为了减少训练开始时的遗忘规模，我们在忘记门的偏差上添加了ignore_bias（默认值：1）。
  Args:
    inputs: input Tensor, 4D, batch x height x width x channels.
    state: state Tensor, 4D, batch x height x width x channels.
    num_channels: the number of output channels in the layer.
    图层中输出通道的数量
    filter_size: the shape of the each convolution filter.
    过滤器大小
    forget_bias: the initial value of the forget biases.
    遗忘偏差
    scope: Optional scope for variable_scope.
    可选范围
    reuse: whether or not the layer and the variables should be reused.
    是否重用图层和变量
  Returns:
     a tuple of tensors representing output and the new state.
     代表输出和新状态的张量的元组
  """
  spatial_size = inputs.get_shape()[1:3]
  if state is None:
    state = init_state(inputs, list(spatial_size) + [2 * num_channels])
  with tf.variable_scope(scope,
                         'BasicConvLstmCell',
                         [inputs, state],
                         reuse=reuse):
    inputs.get_shape().assert_has_rank(4)
    state.get_shape().assert_has_rank(4)
    c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
    inputs_h = tf.concat(axis=3, values=[inputs, h])
    # Parameters of gates are concatenated into one conv for efficiency.
    #为了提高效率，将Gates的参数串联到一个conv中。
    i_j_f_o = layers.conv2d(inputs_h,
                            4 * num_channels, [filter_size, filter_size],
                            stride=1,
                            activation_fn=None,
                            scope='Gates')

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o)

    new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
    new_h = tf.tanh(new_c) * tf.sigmoid(o)

    return new_h, tf.concat(axis=3, values=[new_c, new_h])



