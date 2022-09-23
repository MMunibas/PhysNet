from unittest import TestCase
import numpy as np
from training.DataContainer import DataContainer
from training.DataProvider import DataProvider

class TestDataProvider(TestCase):
    def test_dataprovider(self):
        data = DataContainer('sn2_reactions.npz')
        data_provider = DataProvider(data, 40000, 500,
                             16, 4, 42)
        assert data_provider is not None
