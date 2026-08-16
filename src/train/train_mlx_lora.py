"""
Apple MLX QLoRA Fine-Tuning Pipeline for M4 Apple Silicon
Allows fine-tuning 1B–3B parameter models (Qwen 2.5 1.5B / Llama 3.2 1B) on 16GB Mac Mini with high reasoning quality.
"""
import os
import json
import argparse
import subprocess


def prepare_mlx_dataset(sft_jsonl_in: str, output_dir: str = "data/mlx_sft"):
    """Prepares standard chat JSONL into MLX-LM format."""
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.jsonl")
    valid_file = os.path.join(output_dir, "valid.jsonl")

    records = []
    with open(sft_jsonl_in, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                instr = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                out = item.get("output", "").strip()
                user_msg = f"{instr}\n{inp}".strip() if inp else instr
                
                # Format as messages array
                messages = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": out}
                ]
                records.append({"messages": messages})

    # 90/10 split
    split_idx = max(1, int(len(records) * 0.9))
    train_data = records[:split_idx]
    valid_data = records[split_idx:] if len(records) > 1 else records

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for r in valid_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[✓] MLX SFT dataset created in {output_dir}/")
    print(f"    - Train: {len(train_data)} samples")
    print(f"    - Valid: {len(valid_data)} samples")
    return output_dir


def print_mlx_commands(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", data_dir: str = "data/mlx_sft"):
    print("\n" + "="*60)
    print("   APPLE MLX QLoRA FINE-TUNING COMMANDS FOR M4 MAC MINI   ")
    print("="*60)
    print("\n1. Run Fine-Tuning (Runs directly on M4 GPU & Unified Memory):")
    print(f"python3 -m mlx_lm.lora \\")
    print(f"    --model {model_id} \\")
    print(f"    --data {data_dir} \\")
    print(f"    --train \\")
    print(f"    --iters 600 \\")
    print(f"    --batch-size 2 \\")
    print(f"    --lora-layers 16 \\")
    print(f"    --learning-rate 1e-4 \\")
    print(f"    --adapter-path checkpoints/mlx_adapters")

    print("\n2. Test Fine-Tuned Model in Terminal:")
    print(f"python3 -m mlx_lm.generate \\")
    print(f"    --model {model_id} \\")
    print(f"    --adapter-path checkpoints/mlx_adapters \\")
    print(f"    --prompt '<|im_start|>user\\nTranslate to Tamil: Knowledge is power<|im_end|>\\n<|im_start|>assistant\\n' \\")
    print(f"    --max-tokens 100")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Apple MLX QLoRA Fine-tuning")
    parser.add_argument("--sft_file", default="data/sft/instruct_tamil.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    if os.path.exists(args.sft_file):
        data_dir = prepare_mlx_dataset(args.sft_file)
        print_mlx_commands(args.model, data_dir)
    else:
        print(f"[!] SFT file {args.sft_file} not found.")
