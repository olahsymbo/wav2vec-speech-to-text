import unittest

from wav2vec_speech_to_text.data.loader import MFCCDataset


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = MFCCDataset("testing", n_mfcc=40, input_time_dim=81)

    def test_length(self):
        self.assertGreater(len(self.dataset), 0)

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertEqual(len(sample), 2)
        self.assertEqual(sample[0].shape, (40, 81))
        self.assertIsInstance(sample[1], int)
