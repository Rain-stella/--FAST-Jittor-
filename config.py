"""Configuration for FAST tokenizer."""

from dataclasses import dataclass


@dataclass
class FASTConfig:
    """Hyperparameters for FAST tokenizer."""

    # DCT quantization scale (gamma in paper)
    gamma: float = 10.0

    # BPE vocabulary size
    vocab_size: int = 1024

    # Minimum frequency for BPE (merge frequency threshold)
    min_frequency: int = 2

    # Special tokens (optional)
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"

    # Quantile range for normalization (1st and 99th percentiles)
    q_low: float = 0.01
    q_high: float = 0.99

    # Action chunk length (H) - must be known during tokenization
    chunk_length: int = 50  # e.g., 1 second at 50Hz

    # Action dimensionality (D)
    action_dim: int = 7   # e.g., joint positions