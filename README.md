# FAST: Frequency-space Action Sequence Tokenization (Jittor Implementation)

This repository provides a Jittor-compatible implementation of the FAST tokenizer as described in the paper:
**"FAST: Efficient Action Tokenization for Vision-Language-Action Models"**.

The tokenizer uses Discrete Cosine Transform (DCT) followed by Byte-Pair Encoding (BPE) to compress robot action trajectories into a compact sequence of discrete tokens, enabling efficient training of autoregressive VLA models.

## Features
- Pure Python/NumPy/Scipy core – easily integrable with Jittor.
- Modular design: DCT, BPE, and tokenizer classes.
- Configurable hyperparameters (gamma, vocabulary size, etc.).
- Fit, tokenize, detokenize, save, and load.
- Includes training and evaluation scripts.

## Requirements
- Python >= 3.8
- numpy
- scipy
- tokenizers (HuggingFace)
- jittor (optional, only if you want to use it in a Jittor training pipeline)

Install dependencies:
```bash
pip install numpy scipy tokenizers jittor