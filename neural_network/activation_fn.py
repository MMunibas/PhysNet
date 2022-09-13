import tensorflow as tf
import numpy as np


def swish(x):
    '''google's swish function'''
    return x*tf.sigmoid(x)


def _softplus(x):
    '''First time softplus was used as activation function: "Incorporating Second-Order Functional Knowledge for Better Option Pricing"
    (https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing.pdf)'''
    return tf.log1p(tf.exp(x))


def softplus(x):
    '''this definition is for numerical stability for x larger than 15 (single precision) 
    or x larger than 34 (double precision), there is no numerical difference anymore 
    between the softplus and a linear function'''
    return tf.where(x < 15.0, _softplus(tf.where(x < 15.0, x, tf.zeros_like(x))), x)


def shifted_softplus(x):
    '''return softplus(x) - np.log(2.0)'''
    return tf.nn.softplus(x) - tf.log(2.0)


def scaled_shifted_softplus(x):
    '''this ensures that the function is close to linear near the origin!'''
    return 2*shifted_softplus(x)


def self_normalizing_shifted_softplus(x):
    '''is not really self-normalizing sadly...'''
    return 1.875596256135042*shifted_softplus(x)


def smooth_ELU(x):
    '''general: log((exp(alpha)-1)*exp(x)+1)-alpha'''
    return tf.log1p(1.718281828459045*tf.exp(x))-1.0  # (e-1) = 1.718281828459045


def self_normalizing_smooth_ELU(x):
    return 1.574030675714671*smooth_ELU(x)


def self_normalizing_asinh(x):
    return 1.256734802399369*tf.asinh(x)


def self_normalizing_tanh(x):
    return 1.592537419722831*tf.tanh(x)
