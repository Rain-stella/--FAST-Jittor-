"""
Evaluate FAST tokenizer and generate plots
for comparing PyTorch official version and Jittor version migration results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tokenizer import FASTTokenizer
from config import FASTConfig

# No Chinese fonts needed - all labels are in English
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """Load DROID test data."""
    parquet_path = "./data/droid_100/tasks.parquet"
    df = pd.read_parquet(parquet_path)
    
    action_sequences = []
    for episode_id, group in df.groupby("episode_index"):
        if "frame_index" in df.columns:
            group = group.sort_values("frame_index")
        actions = np.stack(group["action"].values)
        action_sequences.append(actions)
    
    # Split into 50-frame chunks
    H = 50
    chunks = []
    for seq in action_sequences:
        for start in range(0, len(seq) - H + 1, H // 2):
            chunks.append(seq[start:start + H])
    
    return chunks


def evaluate_tokenizer(tokenizer, dataset, num_samples=200):
    """
    Evaluate tokenizer reconstruction quality and compression ratio.
    """
    mse_list = []
    token_len_list = []
    raw_len_list = []
    
    samples = dataset[:num_samples]
    
    for i, sample in enumerate(samples):
        tokens = tokenizer.tokenize(sample)
        reconstructed = tokenizer.detokenize(tokens)
        
        mse = np.mean((sample - reconstructed) ** 2)
        mse_list.append(mse)
        token_len_list.append(len(tokens))
        raw_len_list.append(sample.shape[0] * sample.shape[1])
        
        if (i + 1) % 50 == 0:
            print(f"Evaluated {i+1}/{num_samples} samples")
    
    compression_ratio = np.array(raw_len_list) / np.array(token_len_list)
    
    return {
        'mse_list': mse_list,
        'token_len_list': token_len_list,
        'compression_ratio': compression_ratio,
        'avg_mse': np.mean(mse_list),
        'std_mse': np.std(mse_list),
        'avg_compression': np.mean(compression_ratio),
        'avg_tokens': np.mean(token_len_list)
    }


def plot_reconstruction_comparison(tokenizer, sample, save_dir="./plots"):
    """Plot original vs reconstructed action comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    tokens = tokenizer.tokenize(sample)
    reconstructed = tokenizer.detokenize(tokens)
    
    H, D = sample.shape
    
    fig, axes = plt.subplots(D, 1, figsize=(12, 3 * D))
    if D == 1:
        axes = [axes]
    
    for d in range(D):
        ax = axes[d]
        ax.plot(sample[:, d], 'b-', label='Original', alpha=0.8, linewidth=2)
        ax.plot(reconstructed[:, d], 'r--', label='Reconstructed', alpha=0.8, linewidth=2)
        mse_d = np.mean((sample[:, d] - reconstructed[:, d]) ** 2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Dimension {d+1}')
        ax.set_title(f'Dimension {d+1} Reconstruction (MSE: {mse_d:.6f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_comparison.png", dpi=150)
    plt.close()
    print(f"✅ Reconstruction plot saved to {save_dir}/reconstruction_comparison.png")


def plot_metrics(results, save_dir="./plots"):
    """Plot evaluation metric distributions."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: MSE distribution
    axes[0, 0].hist(results['mse_list'], bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].axvline(results['avg_mse'], color='red', linestyle='--', 
                       label=f'Mean: {results["avg_mse"]:.4f}')
    axes[0, 0].set_xlabel('Reconstruction MSE')
    axes[0, 0].set_ylabel('Samples')
    axes[0, 0].set_title(f'MSE Distribution (Std: {results["std_mse"]:.4f})')
    axes[0, 0].legend()
    
    # Plot 2: Compression ratio distribution
    axes[0, 1].hist(results['compression_ratio'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].axvline(results['avg_compression'], color='red', linestyle='--', 
                       label=f'Mean: {results["avg_compression"]:.2f}x')
    axes[0, 1].set_xlabel('Compression Ratio (raw/tokens)')
    axes[0, 1].set_ylabel('Samples')
    axes[0, 1].set_title('Compression Ratio Distribution')
    axes[0, 1].legend()
    
    # Plot 3: Token count distribution
    axes[1, 0].hist(results['token_len_list'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].axvline(results['avg_tokens'], color='red', linestyle='--',
                       label=f'Mean: {results["avg_tokens"]:.1f}')
    axes[1, 0].set_xlabel('Token Count')
    axes[1, 0].set_ylabel('Samples')
    axes[1, 0].set_title('Token Count Distribution')
    axes[1, 0].legend()
    
    # Plot 4: MSE vs Token Count
    axes[1, 1].scatter(results['token_len_list'], results['mse_list'], alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Token Count')
    axes[1, 1].set_ylabel('Reconstruction MSE')
    axes[1, 1].set_title('MSE vs Token Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_distribution.png", dpi=150)
    plt.close()
    print(f"✅ Metrics distribution saved to {save_dir}/metrics_distribution.png")


def scan_gamma(dataset, gammas=[1, 5, 10, 20, 50, 100]):
    """Scan different gamma values and their effect on MSE and compression ratio."""
    results = []
    
    for gamma in gammas:
        print(f"\n🔬 Testing gamma={gamma}...")
        config = FASTConfig(
            gamma=gamma,
            vocab_size=1024,
            chunk_length=50,
            action_dim=dataset[0].shape[1]
        )
        tokenizer = FASTTokenizer(config)
        tokenizer.fit(dataset[:500])  # quick training
        
        eval_results = evaluate_tokenizer(tokenizer, dataset[:100], num_samples=100)
        
        results.append({
            'gamma': gamma,
            'avg_mse': eval_results['avg_mse'],
            'avg_compression': eval_results['avg_compression'],
            'avg_tokens': eval_results['avg_tokens']
        })
        
        print(f"  gamma={gamma}: MSE={eval_results['avg_mse']:.4f}, Compression={eval_results['avg_compression']:.2f}x")
    
    return results


def plot_gamma_scan(gamma_results, save_dir="./plots"):
    """Plot gamma scan curves."""
    os.makedirs(save_dir, exist_ok=True)
    
    gammas = [r['gamma'] for r in gamma_results]
    mses = [r['avg_mse'] for r in gamma_results]
    compressions = [r['avg_compression'] for r in gamma_results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gamma vs MSE
    axes[0].plot(gammas, mses, 'o-', color='blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('Gamma')
    axes[0].set_ylabel('Avg. Reconstruction MSE')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Gamma vs Reconstruction MSE')
    
    # Gamma vs Compression Ratio
    axes[1].plot(gammas, compressions, 'o-', color='green', linewidth=2, markersize=8)
    axes[1].set_xlabel('Gamma')
    axes[1].set_ylabel('Compression Ratio')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Gamma vs Compression Ratio')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gamma_scan.png", dpi=150)
    plt.close()
    print(f"✅ Gamma scan saved to {save_dir}/gamma_scan.png")


def main():
    print("=" * 60)
    print("FAST Tokenizer Evaluation and Visualization")
    print("=" * 60)
    
    # 1. Load data
    print("\n📂 Loading test data...")
    dataset = load_data()
    print(f"  Total {len(dataset)} action chunks")
    
    # 2. Load trained tokenizer
    print("\n📦 Loading trained tokenizer...")
    tokenizer = FASTTokenizer.load("./fast_tokenizer_droid")
    print("  ✅ Loaded successfully")
    
    # 3. Evaluate tokenizer
    print("\n🔍 Evaluating tokenizer...")
    results = evaluate_tokenizer(tokenizer, dataset, num_samples=200)
    print(f"\n📊 Evaluation Results:")
    print(f"  Average MSE: {results['avg_mse']:.6f} ± {results['std_mse']:.6f}")
    print(f"  Average Compression Ratio: {results['avg_compression']:.2f}x")
    print(f"  Average Token Count: {results['avg_tokens']:.1f}")
    
    # 4. Plot metrics distribution
    print("\n📈 Plotting metrics distribution...")
    plot_metrics(results)
    
    # 5. Plot reconstruction comparison (first sample)
    print("\n📊 Plotting reconstruction comparison...")
    plot_reconstruction_comparison(tokenizer, dataset[0])
    
    # 6. Scan gamma values (optional, time-consuming)
    print("\n🔬 Scanning gamma values (takes ~3-5 minutes)...")
    gamma_results = scan_gamma(dataset, gammas=[1, 5, 10, 20, 50])
    plot_gamma_scan(gamma_results)
    
    print("\n" + "=" * 60)
    print("✅ All plots generated!")
    print(f"  Save directory: ./plots/")
    print("\n📋 Generated files:")
    print("  - reconstruction_comparison.png   (Reconstruction comparison)")
    print("  - metrics_distribution.png        (Metrics distribution)")
    print("  - gamma_scan.png                 (Gamma scan)")
    print("=" * 60)


if __name__ == "__main__":
    main()