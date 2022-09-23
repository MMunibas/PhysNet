from unittest import TestCase
import numpy as np
from training.DataContainer import DataContainer

class TestDataContainer(TestCase):
    def test_datacontainer(self):
        data = DataContainer('sn2_reactions.npz')
        assert data is not None
