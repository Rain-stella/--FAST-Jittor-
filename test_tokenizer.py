"""
Unit tests for FAST tokenizer.
"""

import unittest
import numpy as np
from config import FASTConfig
from tokenizer import FASTTokenizer


class TestFASTTokenizer(unittest.TestCase):
    def setUp(self):
        self.config = FASTConfig(
            gamma=10.0,
            vocab_size=256,   # small for testing
            chunk_length=10,
            action_dim=3,
            min_frequency=1
        )
        # Create dummy dataset
        H = self.config.chunk_length
        D = self.config.action_dim
        self.dataset = []
        for _ in range(50):
            t = np.linspace(0, 2 * np.pi, H)
            actions = np.zeros((H, D))
            for d in range(D):
                actions[:, d] = 0.5 * np.sin(t + d * 0.5) + 0.2 * d
            self.dataset.append(actions)
        self.tokenizer = FASTTokenizer(self.config)

    def test_fit_and_reconstruction(self):
        self.tokenizer.fit(self.dataset)
        sample = self.dataset[0]
        tokens = self.tokenizer.tokenize(sample)
        reconstructed = self.tokenizer.detokenize(tokens)
        mse = np.mean((sample - reconstructed) ** 2)
        # Should be reasonably small (gamma=10 gives some loss)
        self.assertLess(mse, 0.2, "Reconstruction error too high")

    def test_save_load(self):
        self.tokenizer.fit(self.dataset)
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tokenizer.save(tmpdir)
            loaded = FASTTokenizer.load(tmpdir)
            sample = self.dataset[1]
            t1 = self.tokenizer.tokenize(sample)
            t2 = loaded.tokenize(sample)
            self.assertListEqual(t1, t2, "Tokens differ after loading")
            r1 = self.tokenizer.detokenize(t1)
            r2 = loaded.detokenize(t2)
            np.testing.assert_allclose(r1, r2, atol=1e-5)

    def test_compression(self):
        self.tokenizer.fit(self.dataset)
        sample = self.dataset[0]
        tokens = self.tokenizer.tokenize(sample)
        raw_len = self.config.chunk_length * self.config.action_dim
        self.assertLess(len(tokens), raw_len, "No compression achieved")


if __name__ == "__main__":
    unittest.main()