import tensorflow as tf 
import numpy as np 
from neural_network.activation_fn import swish, softplus, shifted_softplus, scaled_shifted_softplus

def test_swish():
    test_input = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    out = swish(test_input)
    with tf.Session() as sess:
        val = out.eval()
    expected = np.array([[0.7310586, 1.7615942], [2.8577223, 3.928055 ]], dtype=np.float32)
    np.testing.assert_array_equal(val, expected)

def test_softplus():
    test_input = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    out = softplus(test_input)
    with tf.Session() as sess:
        val = out.eval()
    expected = np.array([[1.3132616, 2.126928 ], [3.0485873, 4.01815  ]], dtype=np.float32)
    np.testing.assert_array_equal(val, expected)

def test_shifted_softplus():
    test_input = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    out = shifted_softplus(test_input)
    with tf.Session() as sess:
        val = out.eval()
    expected = np.array([[0.62011445, 1.4337809 ],[2.3554401 , 3.3250027 ]], dtype=np.float32)
    np.testing.assert_array_equal(val, expected)
    