import tensorflow as tf
import numpy as np
from .AMSGrad import AMSGrad

class Trainer:
    def __init__(self, learning_rate=1e-3, decay_steps=100000, decay_rate=0.96, scope=None):
        self._scope = scope
        with tf.variable_scope(self.scope):
            self._global_step   = tf.Variable(0, name='global_step', trainable=False)
            self._learning_rate = tf.train.exponential_decay(learning_rate, self._global_step, decay_steps, decay_rate)
            self._optimizer     = AMSGrad(learning_rate=self._learning_rate)

    def build_train_op(self, loss, moving_avg_decay=0.999, max_norm=10.0, dependencies=[]):        
        #clipped gradients
        gradients, variables = zip(*self._optimizer.compute_gradients(loss))
        summary_op = tf.summary.scalar("global_gradient_norm", tf.global_norm(gradients))
        gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
        apply_gradient_op = self._optimizer.apply_gradients(zip(gradients, variables), global_step=self._global_step)
        
        #get model variable collection
        self._model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        #create ExponentialMovingAverage object and build its apply operation
        self._ema = tf.train.ExponentialMovingAverage(moving_avg_decay, self._global_step)
        ema_op = self.ema.apply(self.model_vars)

        #make backup variables
        with tf.variable_scope('backup_variables'):
            self._backup_vars = [tf.get_variable(var.op.name, dtype=var.value().dtype, trainable=False, 
                initializer=var.initialized_value()) for var in self.model_vars]

        #generate train op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(
                            [apply_gradient_op,
                             summary_op,
                             ema_op] + update_ops + dependencies):
            train_op = tf.no_op(name='train')

        return train_op

    def load_averaged_variables(self):
        return tf.group(*(tf.assign(var, self.ema.average(var).read_value())
                         for var in self.model_vars))

    def save_variable_backups(self):
        return tf.group(*(tf.assign(bck, var.read_value())
                         for var, bck in zip(self.model_vars, self.backup_vars)))

    def restore_variable_backups(self):
        return tf.group(*(tf.assign(var, bck.read_value())
                         for var, bck in zip(self.model_vars, self.backup_vars)))

    @property
    def scope(self):
        return self._scope

    @property
    def global_step(self):
        return self._global_step 

    @property
    def ema(self):
        return self._ema

    @property
    def model_vars(self):
        return self._model_vars

    @property
    def backup_vars(self):
        return self._backup_vars
    
    
       
