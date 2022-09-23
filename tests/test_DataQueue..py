from unittest import TestCase
import numpy as np
from training.DataContainer import DataContainer
from training.DataProvider import DataProvider
from training.DataQueue import DataQueue

class TestDataQueue(TestCase):
    def test_dataqueue(self):
        data = DataContainer('sn2_reactions.npz')
        data_provider = DataProvider(data, 40000, 500,
                             16, 4, 42)
        data_queue = DataQueue(data_provider.next_batch,
                             capacity=1000, scope="train_data_queue")
        assert data_queue is not None

