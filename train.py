import os
import numpy as np
import pandas as pd
from config import FASTConfig
from tokenizer import FASTTokenizer


def load_demo_data():
    """
    直接读取本地 DROID 数据（使用明确路径）
    """
    # 直接指定路径
    parquet_path = "./data/droid_100/tasks.parquet"
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"找不到数据文件: {parquet_path}")
    
    print(f"读取数据文件: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"数据列: {df.columns.tolist()}")
    print(f"总帧数: {len(df)}")
    
    # 按 episode 分组提取动作
    action_sequences = []
    for episode_id, group in df.groupby("episode_index"):
        if "frame_index" in df.columns:
            group = group.sort_values("frame_index")
        actions = np.stack(group["action"].values)
        action_sequences.append(actions)
    
    print(f"提取了 {len(action_sequences)} 个 episode")
    action_dim = action_sequences[0].shape[1]
    print(f"动作维度: {action_dim}")
    
    # 切分为固定长度片段（50帧）
    H = 50
    chunks = []
    for seq in action_sequences:
        for start in range(0, len(seq) - H + 1, H // 2):
            chunks.append(seq[start:start + H])
    
    print(f"切分为 {len(chunks)} 个动作片段，每个形状: (50, {action_dim})")
    return chunks


def main():
    print("加载 DROID 数据集...")
    dataset = load_demo_data()
    print(f"共 {len(dataset)} 个训练样本")
    
    if not dataset:
        raise ValueError("数据集为空")
    action_dim = dataset[0].shape[1]
    
    config = FASTConfig(
        gamma=10.0,
        vocab_size=1024,
        chunk_length=50,
        action_dim=action_dim
    )

    tokenizer = FASTTokenizer(config)
    print("训练 FAST tokenizer...")
    tokenizer.fit(dataset)

    save_dir = "./fast_tokenizer_droid"
    tokenizer.save(save_dir)
    print(f"Tokenizer 已保存到 {save_dir}")

    sample = dataset[0]
    tokens = tokenizer.tokenize(sample)
    reconstructed = tokenizer.detokenize(tokens)
    mse = np.mean((sample - reconstructed) ** 2)
    print(f"样本重建 MSE: {mse:.6f}")
    print(f"原始长度: {sample.shape[0] * sample.shape[1]}, Token 数: {len(tokens)}")


if __name__ == "__main__":
    main()