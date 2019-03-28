import tensorflow as tf
from .NeuronLayer   import *
from .DenseLayer    import *
from .ResidualLayer import *

class OutputBlock(NeuronLayer):
    def __str__(self):
        return "output"+super().__str__()

    def __init__(self, F, num_residual, activation_fn=None, seed=None, scope=None, keep_prob=1.0, dtype=tf.float32):
        super().__init__(F, 2, activation_fn)
        with tf.variable_scope(scope):
            self._residual_layer = []
            for i in range(num_residual):
                self._residual_layer.append(ResidualLayer(F, F, activation_fn, seed=seed, scope="residual_layer"+str(i), keep_prob=keep_prob, dtype=dtype))
            self._dense = DenseLayer(F, 2, W_init=tf.zeros([F, 2], dtype=dtype), use_bias=False, scope="dense_layer", dtype=dtype)
    
    @property
    def residual_layer(self):
        return self._residual_layer

    @property
    def dense(self):
        return self._dense

    def __call__(self, x):
        for i in range(len(self.residual_layer)):
            x = self.residual_layer[i](x)
        if self.activation_fn is not None: 
            x = self.activation_fn(x)
        return self.dense(x)
