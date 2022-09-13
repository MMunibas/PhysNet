from unittest import TestCase
from NNCalculator import NNCalculator
import pickle
import numpy as np

class TestNNCalculator(TestCase):
    def test_get_potential_energy(self):
        atom = pickle.load(open('tests/atom.pkl', 'rb'))
        calc = NNCalculator(checkpoint='tests/checkpoint/best_model.ckpt', atoms=atom)
        assert calc.get_potential_energy(atom)[0] == np.float32(-13.807366)