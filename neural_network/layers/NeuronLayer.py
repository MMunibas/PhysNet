import tensorflow as tf

''' parent class for all neuron layers '''
class NeuronLayer:
    def __str__(self):
        return "[" + str(self.n_in) + "->" + str(self.n_out) + "]"

    def __init__(self, n_in, n_out, activation_fn=None):
        self._n_in  = n_in  #number of inputs
        self._n_out = n_out #number of outpus
        self._activation_fn = activation_fn #activation function
            
    @property
    def n_in(self):
        return self._n_in
    
    @property
    def n_out(self):
        return self._n_out
    
    @property
    def activation_fn(self):
        return self._activation_fn