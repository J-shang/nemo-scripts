import argparse
import torch
import os
from megatron.core.datasets.indexed_dataset import IndexedDataset

# 如果你有 tokenizer，可以在这里加载以查看真实文本
# from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-prefix', type=str, required=True, 
                        help='不带后缀的文件路径前缀，例如 "my_data_text_document"')
    args = parser.parse_args()

    # 1. 检查文件是否存在
    bin_file = args.data_prefix + '.bin'
    idx_file = args.data_prefix + '.idx'
    
    if not os.path.exists(bin_file) or not os.path.exists(idx_file):
        print(f"Error: 找不到 {bin_file} 或 {idx_file}")
        return

    # 2. 加载 Dataset
    # impl='mmap' 是最常用的方式，速度快且不占内存
    ds = IndexedDataset(args.data_prefix)

    print(f"--> 数据集总样本数: {len(ds)}")
    print("-" * 30)

    # 3. 打印前 5 条样本
    num_samples = min(5, len(ds))
    
    # Optional: 加载 Tokenizer (以 Qwen/Llama 为例)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    
    for i in range(num_samples):
        # 获取第 i 条数据的 token ids (numpy array)
        token_ids = ds[i]
        
        print(f"\nSample {i}:")
        print(f"  Length: {len(token_ids)}")
        print(f"  Raw Token IDs (前20个): {token_ids[:20]} ...")
        
        # 如果你想看解码后的文本 (取消注释)
        print(f"  Decoded Text: {tokenizer.decode(token_ids)}")

if __name__ == "__main__":
    main()