from unittest import TestCase
from training.Trainer import Trainer

class TestTrainer(TestCase):
    def test_trainer(self):
        trainer = Trainer(1e-3, 10000000,
                  0.01, scope="trainer")
        assert trainer is not None
