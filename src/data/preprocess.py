"""
Data Preprocessing & Binary Dataset Builder for Tamil Nano LLM
Converts text corpora and SFT JSONL files into high-throughput memory-mapped binary shards (uint16/uint32).
"""
import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer


def build_pretrain_bin(
    input_file: str,
    tokenizer_path: str,
    output_dir: str,
    val_split_ratio: float = 0.05,
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|eos|>") or 2
    bos_id = tokenizer.token_to_id("<|bos|>") or 1

    print(f"[*] Reading and tokenizing pretraining text from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        paragraphs = f.read().split("\n\n")

    all_tokens = []
    for para in tqdm(paragraphs, desc="Tokenizing"):
        para = para.strip()
        if not para:
            continue
        ids = tokenizer.encode(para).ids
        if ids:
            all_tokens.append(bos_id)
            all_tokens.extend(ids)
            all_tokens.append(eos_id)

    total_tokens = len(all_tokens)
    print(f"[+] Total pretraining tokens collected: {total_tokens:,}")

    if total_tokens == 0:
        raise ValueError("No tokens found. Please check input file content.")

    # Split Train / Validation
    val_size = int(total_tokens * val_split_ratio)
    train_tokens = all_tokens[:-val_size] if val_size > 0 else all_tokens
    val_tokens = all_tokens[-val_size:] if val_size > 0 else all_tokens[:100]

    # Save as memory-mapped uint16 binary files (or uint32 if vocab > 65535)
    dtype = np.uint16 if tokenizer.get_vocab_size() <= 65535 else np.uint32
    
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    np.array(train_tokens, dtype=dtype).tofile(train_path)
    np.array(val_tokens, dtype=dtype).tofile(val_path)

    print(f"[✓] Saved binary token shards:")
    print(f"    - Train: {train_path} ({len(train_tokens):,} tokens, {os.path.getsize(train_path) / 1024:.1f} KB)")
    print(f"    - Val  : {val_path} ({len(val_tokens):,} tokens, {os.path.getsize(val_path) / 1024:.1f} KB)")


def build_sft_dataset(
    sft_jsonl_file: str,
    tokenizer_path: str,
    output_file: str,
    max_length: int = 1024,
):
    """Tokenizes and creates ChatML instruction-following dataset."""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<|pad|>") or 0
    eos_id = tokenizer.token_to_id("<|eos|>") or 2
    bos_id = tokenizer.token_to_id("<|bos|>") or 1

    print(f"[*] Preparing SFT dataset from: {sft_jsonl_file}")
    records = []
    with open(sft_jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    processed_samples = []
    for item in tqdm(records, desc="Processing SFT"):
        instruction = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        output = item.get("output", "").strip()

        user_content = f"{instruction}\n{inp}".strip() if inp else instruction
        
        # Format as ChatML
        # <|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>
        prompt_text = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        full_text = prompt_text + f"{output}<|im_end|>"

        prompt_ids = tokenizer.encode(prompt_text).ids
        full_ids = tokenizer.encode(full_text).ids

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        # In SFT, we mask loss on user prompt (set label to -100 / pad)
        # so the model only learns to predict the assistant's response
        input_ids = full_ids
        labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]

        processed_samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "task": item.get("task", "general")
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for s in processed_samples:
            f.write(json.dumps(s) + "\n")

    print(f"[✓] Saved SFT dataset with {len(processed_samples)} samples to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and pack Tamil datasets")
    parser.add_argument("--pretrain_file", default="data/raw/pretrain_tamil.txt")
    parser.add_argument("--sft_file", default="data/sft/instruct_tamil.jsonl")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer/tokenizer.json")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    if os.path.exists(args.pretrain_file) and os.path.exists(args.tokenizer):
        build_pretrain_bin(args.pretrain_file, args.tokenizer, args.out_dir)

    if os.path.exists(args.sft_file) and os.path.exists(args.tokenizer):
        sft_out = os.path.join(args.out_dir, "sft_processed.jsonl")
        build_sft_dataset(args.sft_file, args.tokenizer, sft_out)
