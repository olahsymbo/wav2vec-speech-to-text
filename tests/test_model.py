import unittest

import torch

from wav2vec_speech_to_text.model.cnn import MFCC_CNN


class TestCNN(unittest.TestCase):
    def setUp(self):
        self.model = MFCC_CNN(num_classes=10)

    def test_initialization(self):
        self.assertIsInstance(self.model, MFCC_CNN)
        self.assertEqual(self.model.fc2.out_features, 10)

    def test_forward_shape(self):
        sample_input = torch.randn(4, 40, 81)
        output = self.model(sample_input)
        self.assertEqual(output.shape, (4, 10))
