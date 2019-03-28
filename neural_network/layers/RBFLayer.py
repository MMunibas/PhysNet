import tensorflow as tf
import numpy as np
from .NeuronLayer import *
from .DenseLayer import *

#inverse softplus transformation
def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))

#radial basis function expansion
class RBFLayer(NeuronLayer):
    def __str__(self):
        return "radial_basis_function_layer"+super().__str__()

    def __init__(self, K, cutoff, scope=None, dtype=tf.float32):
        super().__init__(1, K, None)
        self._K = K
        self._cutoff = cutoff
        with tf.variable_scope(scope):
            #initialize centers
            centers = softplus_inverse(np.linspace(1.0,np.exp(-cutoff),K))
            self._centers = tf.nn.softplus(tf.Variable(np.asarray(centers), name="centers", dtype=dtype))
            tf.summary.histogram("rbf_centers", self.centers) 

            #initialize widths (inverse softplus transformation is applied, such that softplus can be used to guarantee positive values)
            widths = [softplus_inverse((0.5/((1.0-np.exp(-cutoff))/K))**2)]*K
            self._widths = tf.nn.softplus(tf.Variable(np.asarray(widths),  name="widths",  dtype=dtype))
            tf.summary.histogram("rbf_widths", self.widths)

    @property
    def K(self):
        return self._K

    @property
    def cutoff(self):
        return self._cutoff
    
    @property
    def centers(self):
        return self._centers   

    @property
    def widths(self):
        return self._widths  

    #cutoff function that ensures a smooth cutoff
    def cutoff_fn(self, D):
        x = D/self.cutoff
        x3 = x**3
        x4 = x3*x
        x5 = x4*x
        return tf.where(x < 1, 1 - 6*x5 + 15*x4 - 10*x3, tf.zeros_like(x))
    
    def __call__(self, D):
        D = tf.expand_dims(D, -1) #necessary for proper broadcasting behaviour
        rbf = self.cutoff_fn(D)*tf.exp(-self.widths*(tf.exp(-D)-self.centers)**2)
        return rbf



