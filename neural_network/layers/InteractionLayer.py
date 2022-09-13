import tensorflow as tf
import numpy as np
from .RBFLayer import *
from .NeuronLayer import *
from .DenseLayer import *
from .ResidualLayer import *


class InteractionLayer(NeuronLayer):
    def __str__(self):
        return "interaction_layer"+super().__str__()

    def __init__(self, K, F, num_residual, activation_fn=None, seed=None, scope=None, keep_prob=1.0, dtype=tf.float32):
        super().__init__(K, F, activation_fn)
        self._keep_prob = keep_prob
        with tf.variable_scope(scope):
            # transforms radial basis functions to feature space
            self._k2f = DenseLayer(K, F, W_init=tf.zeros(
                [K, F], dtype=dtype), use_bias=False, seed=seed, scope='k2f', dtype=dtype)
            # rearrange feature vectors for computing the "message"
            self._dense_i = DenseLayer(
                F, F, activation_fn, seed=seed, scope="dense_i", dtype=dtype)  # central atoms
            self._dense_j = DenseLayer(
                F, F, activation_fn, seed=seed, scope="dense_j", dtype=dtype)  # neighbouring atoms
            # for performing residual transformation on the "message"
            self._residual_layer = []
            for i in range(num_residual):
                self._residual_layer.append(ResidualLayer(
                    F, F, activation_fn, seed=seed, scope="residual_layer"+str(i), keep_prob=keep_prob, dtype=dtype))
            # for performing the final update to the feature vectors
            self._dense = DenseLayer(
                F, F, seed=seed, scope="dense", dtype=dtype)
            self._u = tf.Variable(
                tf.ones([F], dtype=dtype), name="u", dtype=dtype)
            tf.summary.histogram("gates",  self.u)

    @property
    def keep_prob(self):
        '''keep_prob getter'''
        return self._keep_prob

    @property
    def k2f(self):
        '''k2f getter'''
        return self._k2f

    @property
    def dense_i(self):
        '''dense_i getter'''
        return self._dense_i

    @property
    def dense_j(self):
        '''dense_j getter'''
        return self._dense_j

    @property
    def residual_layer(self):
        '''residual_layer getter'''
        return self._residual_layer

    @property
    def dense(self):
        '''dense getter'''
        return self._dense

    @property
    def u(self):
        '''u getter'''
        return self._u

    def __call__(self, x, rbf, idx_i, idx_j):
        # pre-activation
        if self.activation_fn is not None:
            xa = tf.nn.dropout(self.activation_fn(x), self.keep_prob)
        else:
            xa = tf.nn.dropout(x, self.keep_prob)
        # calculate feature mask from radial basis functions
        g = self.k2f(rbf)
        # calculate contribution of neighbors and central atom
        xi = self.dense_i(xa)
        xj = tf.segment_sum(g*tf.gather(self.dense_j(xa), idx_j), idx_i)
        # add contributions to get the "message"
        m = xi + xj
        for i in range(len(self.residual_layer)):
            m = self.residual_layer[i](m)
        if self.activation_fn is not None:
            m = self.activation_fn(m)
        x = self.u*x + self.dense(m)
        return x
