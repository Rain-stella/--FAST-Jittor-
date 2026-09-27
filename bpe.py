"""
Byte-Pair Encoding (BPE) tokenizer - Pure Python Implementation
No external dependencies (no huggingface tokenizers)
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import os


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer that operates on lists of integers.
    Pure Python implementation, no external dependencies.
    """
    
    def __init__(self, vocab_size: int = 1024, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges: List[Tuple[int, int]] = []           # 合并规则列表
        self.merge_map: Dict[Tuple[int, int], int] = {}   # pair -> merged_token_id 映射
        self.vocab: Dict[str, int] = {}                   # 词汇表: token_str -> token_id
        self.id_to_token: Dict[int, str] = {}             # token_id -> token_str
        self.special_tokens = {"<PAD>": 0, "<UNK>": 1}
        self._is_trained = False

    def _get_pair_stats(self, word: List[int]) -> Dict[Tuple[int, int], int]:
        """统计相邻 token 对的频率"""
        stats = defaultdict(int)
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += 1
        return stats

    def _merge_pair(self, word: List[int], pair: Tuple[int, int], new_token: int) -> List[int]:
        """将 word 中所有相邻的 pair 合并为 new_token"""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word

    def train(self, sequences: List[List[int]], verbose: bool = False):
        """
        训练 BPE 词表
        """
        if not sequences:
            raise ValueError("训练序列不能为空")
        
        # Step 1: 初始化词表：每个数字作为独立 token
        all_numbers = set()
        for seq in sequences:
            all_numbers.update(seq)
        
        next_id = len(self.special_tokens)
        self.vocab = dict(self.special_tokens)
        for num in sorted(all_numbers):
            self.vocab[str(num)] = next_id
            next_id += 1
        
        # 每个序列转换为 token_id 列表
        corpus = []
        for seq in sequences:
            corpus.append([self.vocab[str(x)] for x in seq])
        
        # Step 2: 迭代合并
        current_vocab_size = len(self.vocab)
        target_size = min(self.vocab_size, current_vocab_size + 10000)
        
        self.merges = []
        self.merge_map = {}  # 清空映射
        
        iteration = 0
        while current_vocab_size < target_size and len(corpus) > 0:
            pair_stats = defaultdict(int)
            for word in corpus:
                if len(word) < 2:
                    continue
                stats = self._get_pair_stats(word)
                for pair, count in stats.items():
                    pair_stats[pair] += count
            
            if not pair_stats:
                break
            
            most_frequent_pair = max(pair_stats.items(), key=lambda x: x[1])
            pair, max_count = most_frequent_pair
            
            if max_count < self.min_frequency:
                break
            
            # 创建新 token
            new_token_id = current_vocab_size
            new_token_str = f"<t_{new_token_id}>"
            self.vocab[new_token_str] = new_token_id
            current_vocab_size += 1
            
            # 记录合并规则 和 映射
            self.merges.append(pair)
            self.merge_map[pair] = new_token_id  # 保存 pair -> token_id 映射
            
            # 合并所有词中的该 pair
            new_corpus = []
            for word in corpus:
                new_word = self._merge_pair(word, pair, new_token_id)
                new_corpus.append(new_word)
            corpus = new_corpus
            
            iteration += 1
            if verbose and iteration % 100 == 0:
                print(f"  BPE 迭代 {iteration}: 合并 {pair} -> {new_token_str}, 频率 {max_count}")
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._is_trained = True
        
        if verbose:
            print(f"BPE 训练完成: 词汇量 {len(self.vocab)}, 合并了 {len(self.merges)} 次")

    def encode(self, sequence: List[int]) -> List[int]:
        """将整数序列编码为 token IDs"""
        if not self._is_trained:
            raise RuntimeError("Tokenizer 未训练，请先调用 train()")
        
        # Step 1: 将每个数字转为初始 token
        word = []
        for x in sequence:
            key = str(x)
            if key in self.vocab:
                word.append(self.vocab[key])
            else:
                word.append(self.vocab["<UNK>"])
        
        # Step 2: 逐步应用合并规则（使用 merge_map 查找正确的 token ID）
        for pair in self.merges:
            if pair in self.merge_map:
                new_token_id = self.merge_map[pair]
                word = self._merge_pair(word, pair, new_token_id)
            else:
                # 如果映射中找不到，跳过（理论上不会发生）
                continue
        
        return word

    def decode(self, token_ids: List[int]) -> List[int]:
        """将 token IDs 解码回整数序列"""
        if not self._is_trained:
            raise RuntimeError("Tokenizer 未训练，请先调用 train()")
        
        # 收集所有 token 对应的字符串
        tokens_str = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens_str.append(self.id_to_token[tid])
            else:
                tokens_str.append("<UNK>")
        
        # 展开所有合并 token
        def expand_token(tok_str: str) -> List[int]:
            if tok_str.startswith("<t_"):
                # 合并 token，需要逆向展开
                # 找到对应的合并规则
                token_id = self.vocab.get(tok_str, -1)
                if token_id == -1:
                    return []
                # 遍历 merge_map 找到对应的 pair
                for pair, merged_id in self.merge_map.items():
                    if merged_id == token_id:
                        # 递归展开 pair 中的两个 token
                        # pair 中的元素可能是原始数字 ID，也可能是合并 token ID
                        left_str = self.id_to_token.get(pair[0], "")
                        right_str = self.id_to_token.get(pair[1], "")
                        # 递归展开
                        left_expanded = expand_token(left_str) if left_str else [pair[0]]
                        right_expanded = expand_token(right_str) if right_str else [pair[1]]
                        # 如果 left_str 是以 <t_ 开头的，说明是合并 token
                        if left_str.startswith("<t_"):
                            left_expanded = expand_token(left_str)
                        else:
                            left_expanded = [pair[0]] if pair[0] in self.vocab.values() else []
                        if right_str.startswith("<t_"):
                            right_expanded = expand_token(right_str)
                        else:
                            right_expanded = [pair[1]] if pair[1] in self.vocab.values() else []
                        return left_expanded + right_expanded
                return []
            else:
                # 普通数字 token
                try:
                    return [int(tok_str)]
                except ValueError:
                    return []
        
        # 展开所有 token
        result = []
        for tok_str in tokens_str:
            expanded = expand_token(tok_str)
            if expanded:
                result.extend(expanded)
            else:
                # 如果无法展开，尝试直接解析数字
                try:
                    result.append(int(tok_str))
                except ValueError:
                    pass
        
        return result

    def save(self, path: str):
        """保存 tokenizer 到文件"""
        if not self._is_trained:
            raise RuntimeError("无法保存未训练的 tokenizer")
        
        # 将 merge_map 转换为可序列化的格式（tuple 作为 key 需要转为 list）
        merge_map_serializable = {f"{k[0]},{k[1]}": v for k, v in self.merge_map.items()}
        
        data = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "vocab": self.vocab,
            "merges": self.merges,
            "merge_map": merge_map_serializable,
            "special_tokens": self.special_tokens
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """从文件加载 tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"],
            min_frequency=data["min_frequency"]
        )
        tokenizer.vocab = data["vocab"]
        # 关键修复：将 list 转为 tuple，确保可哈希
        tokenizer.merges = [tuple(pair) for pair in data["merges"]]
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer._is_trained = True

        # 还原 merge_map（键已经是 tuple）
        tokenizer.merge_map = {}
        for k, v in data.get("merge_map", {}).items():
            left, right = map(int, k.split(","))
            tokenizer.merge_map[(left, right)] = v

        return tokenizer

    def get_vocab_size(self) -> int:
        return len(self.vocab) if self.vocab else 0