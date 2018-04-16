import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest


class GRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
  """

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               layer_norm=False):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or tf.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._gate_linear = None
    self._candidate_linear = None
    self.layer_norm = layer_norm

    if self.layer_norm:
        self.gammas = {}
        self.betas = {}
        for gate in ['r', 'u', 'c']:
          self.gammas[gate] = tf.get_variable(
              'gamma_' + gate, shape=[num_units], initializer=tf.constant_initializer(1.0))
          self.betas[gate] = tf.get_variable(
              'beta_' + gate, shape=[num_units], initializer=tf.constant_initializer(0.0))

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    if self._gate_linear is None:
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        bias_ones = tf.constant_initializer(1.0, dtype=inputs.dtype)
      with tf.variable_scope("gates"):  # Reset gate and update gate.
        self._gate_linear = _Linear(
            [inputs, state],
            2 * self._num_units,
            True,
            bias_initializer=bias_ones,
            kernel_initializer=self._kernel_initializer)

    def Norm(inputs, gamma, beta): 
      m, v = tf.nn.moments(inputs, [1], keep_dims=True)
      normalized_input = (inputs - m) / tf.sqrt(v + 1e-5)
      return normalized_input * gamma + beta

    gate_out = self._gate_linear([inputs, state])
    pre_r, pre_u = tf.split(value=gate_out, num_or_size_splits=2, axis=1)
    
    if self.layer_norm:
      pre_r, pre_u = (Norm(pre_r, self.gammas['r'], self.betas['r']),
                      Norm(pre_u, self.gammas['u'], self.gammas['u']))
    r, u = tf.sigmoid(pre_r), tf.sigmoid(pre_u)

    r_state = r * state
    if self._candidate_linear is None:
      with tf.variable_scope("candidate"):
        self._candidate_linear = _Linear(
            [inputs, r_state],
            self._num_units,
            True,
            bias_initializer=self._bias_initializer,
            kernel_initializer=self._kernel_initializer)
    pre_c = self._candidate_linear([inputs, r_state])
    if self.layer_norm:
      pre_c = Norm(pre_c, self.gammas['c'], self.betas['c'])
    c = self._activation(pre_c)
    new_h = u * state + (1 - u) * c
    return new_h, new_h


class _Linear(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
      self._weights = tf.get_variable(
          'kernel', [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with tf.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
          self._biases = tf.get_variable(
              'bias', [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = tf.matmul(args[0], self._weights)
    else:
      res = tf.matmul(tf.concat(args, 1), self._weights)
    if self._build_bias:
      res = tf.nn.bias_add(res, self._biases)
    return res
