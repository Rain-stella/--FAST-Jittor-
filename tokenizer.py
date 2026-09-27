"""
FAST Tokenizer - Jittor compatible implementation.
No Python 3.10+ type annotations (compatible with Python 3.8/3.9).
"""

import os
import pickle
import numpy as np
from typing import List, Optional, Tuple
from config import FASTConfig
from dct import dct, idct
from bpe import BPETokenizer


class FASTTokenizer:
    """
    Frequency-space Action Sequence Tokenizer (FAST).
    Uses DCT + BPE to compress action trajectories into discrete tokens.
    """

    def __init__(self, config: FASTConfig):
        self.config = config
        self.bpe = BPETokenizer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency
        )
        self.low_quantiles = None   # shape (D,)
        self.high_quantiles = None  # shape (D,)
        self.gamma = config.gamma

    def _normalize(self, actions: np.ndarray) -> np.ndarray:
        """Normalize each dimension using precomputed quantiles."""
        if self.low_quantiles is None or self.high_quantiles is None:
            raise RuntimeError("Tokenizer not fitted. Call fit() first.")
        D = actions.shape[1]
        norm = np.zeros_like(actions, dtype=np.float32)
        for d in range(D):
            low = self.low_quantiles[d]
            high = self.high_quantiles[d]
            range_val = high - low
            if range_val < 1e-12:
                norm[:, d] = 0.0
            else:
                norm[:, d] = (actions[:, d] - low) / range_val * 2.0 - 1.0
        return norm

    def _denormalize(self, norm_actions: np.ndarray) -> np.ndarray:
        """Inverse of _normalize."""
        if self.low_quantiles is None or self.high_quantiles is None:
            raise RuntimeError("Tokenizer not fitted.")
        D = norm_actions.shape[1]
        actions = np.zeros_like(norm_actions, dtype=np.float32)
        for d in range(D):
            low = self.low_quantiles[d]
            high = self.high_quantiles[d]
            range_val = high - low
            if range_val < 1e-12:
                actions[:, d] = low
            else:
                actions[:, d] = ((norm_actions[:, d] + 1.0) / 2.0) * range_val + low
        return actions

    def fit(self, action_dataset: List[np.ndarray]):
        """
        Fit the tokenizer: compute quantiles, train BPE on all DCT coefficients.
        """
        # Step 1: compute quantiles from all data
        all_actions = np.concatenate(action_dataset, axis=0)
        self.low_quantiles = np.percentile(all_actions, self.config.q_low * 100, axis=0)
        self.high_quantiles = np.percentile(all_actions, self.config.q_high * 100, axis=0)

        H = self.config.chunk_length
        D = self.config.action_dim
        sequences_for_bpe = []

        for seq in action_dataset:
            if seq.shape != (H, D):
                raise ValueError(f"Expected shape ({H}, {D}), got {seq.shape}")
            norm_seq = self._normalize(seq)
            coeff_matrix = np.zeros_like(norm_seq)
            for d in range(D):
                coeff_matrix[:, d] = dct(norm_seq[:, d], norm='ortho')
            quant_matrix = np.round(self.gamma * coeff_matrix).astype(np.int32)
            # flatten: low-frequency first, interleaving dimensions
            flat = []
            for k in range(H):
                for d in range(D):
                    flat.append(int(quant_matrix[k, d]))
            sequences_for_bpe.append(flat)

        self.bpe.train(sequences_for_bpe, verbose=True)

    def tokenize(self, action_chunk: np.ndarray) -> List[int]:
        """Tokenize a single action chunk (H, D) into token IDs."""
        H, D = action_chunk.shape
        if H != self.config.chunk_length or D != self.config.action_dim:
            raise ValueError(f"Input shape {action_chunk.shape} does not match config")

        norm_seq = self._normalize(action_chunk)
        coeff_matrix = np.zeros_like(norm_seq)
        for d in range(D):
            coeff_matrix[:, d] = dct(norm_seq[:, d], norm='ortho')
        quant_matrix = np.round(self.gamma * coeff_matrix).astype(np.int32)
        flat = []
        for k in range(H):
            for d in range(D):
                flat.append(int(quant_matrix[k, d]))
        tokens = self.bpe.encode(flat)
        return tokens

    def detokenize(self, token_ids: List[int]) -> np.ndarray:
        """Reconstruct action chunk from token IDs."""
        flat_ints = self.bpe.decode(token_ids)
        H = self.config.chunk_length
        D = self.config.action_dim
        if len(flat_ints) != H * D:
            if len(flat_ints) < H * D:
                flat_ints += [0] * (H * D - len(flat_ints))
            else:
                flat_ints = flat_ints[:H * D]
        quant_matrix = np.array(flat_ints, dtype=np.float32).reshape((H, D))
        coeff_matrix = quant_matrix / self.gamma
        rec_norm = np.zeros_like(coeff_matrix)
        for d in range(D):
            rec_norm[:, d] = idct(coeff_matrix[:, d], norm='ortho')
        rec_actions = self._denormalize(rec_norm)
        return rec_actions

    def save(self, save_dir: str):
        """Save tokenizer state to disk."""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        np.save(os.path.join(save_dir, 'low_quantiles.npy'), self.low_quantiles)
        np.save(os.path.join(save_dir, 'high_quantiles.npy'), self.high_quantiles)
        self.bpe.save(os.path.join(save_dir, 'bpe_tokenizer.json'))

    @classmethod
    def load(cls, save_dir: str):
        """Load tokenizer from disk."""
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        tokenizer = cls(config)
        tokenizer.low_quantiles = np.load(os.path.join(save_dir, 'low_quantiles.npy'))
        tokenizer.high_quantiles = np.load(os.path.join(save_dir, 'high_quantiles.npy'))
        tokenizer.bpe = BPETokenizer.load(os.path.join(save_dir, 'bpe_tokenizer.json'))
        return tokenizer