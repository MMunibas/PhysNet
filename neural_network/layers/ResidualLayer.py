import tensorflow as tf
import numpy as np
from .NeuronLayer import *
from .DenseLayer import *

class ResidualLayer(NeuronLayer):
    def __str__(self):
        return "residual_layer"+super().__str__()

    def __init__(self, n_in, n_out, activation_fn=None, W_init=None, b_init=None, use_bias=True, seed=None, scope=None, keep_prob=1.0, dtype=tf.float32):
        super().__init__(n_in, n_out, activation_fn)
        self._keep_prob = keep_prob
        with tf.variable_scope(scope):
            self._dense    = DenseLayer(n_in,  n_out, activation_fn=activation_fn, 
                W_init=W_init, b_init=b_init, use_bias=use_bias, seed=seed, scope="dense", dtype=dtype)
            self._residual = DenseLayer(n_out, n_out, activation_fn=None, 
                W_init=W_init, b_init=b_init, use_bias=use_bias, seed=seed, scope="residual", dtype=dtype)
      
    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def dense(self):
        return self._dense

    @property
    def residual(self):
        return self._residual

    def __call__(self, x):
        #pre-activation
        if self.activation_fn is not None: 
            y = tf.nn.dropout(self.activation_fn(x), self.keep_prob)
        else:
            y = tf.nn.dropout(x, self.keep_prob)
        x += self.residual(self.dense(y))
        return x
