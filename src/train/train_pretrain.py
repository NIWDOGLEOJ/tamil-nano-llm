"""
Pre-training Engine for Tamil Nano/Micro LLM
Optimized for Apple Silicon Metal (MPS), CUDA, and CPU.
Supports Cosine Learning Rate Schedule with Warmup, Gradient Accumulation, and Validation Perplexity Tracking.
"""
import os
import sys
import time
import math
import argparse
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.config import TamilNanoConfig, CONFIG_PRESETS
from src.model.transformer_torch import TamilNanoForCausalLM


class MemoryMappedPretrainDataset(Dataset):
    """Zero-overhead memory-mapped dataset for sequential token chunk streaming."""
    def __init__(self, bin_path: str, block_size: int):
        self.bin_path = bin_path
        self.block_size = block_size
        
        # Open binary file with memmap
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.num_samples = max(0, (self.total_tokens - 1) // self.block_size)

        if self.num_samples == 0:
            # Fallback if corpus is shorter than full block size: pad to block size
            self.data = np.array(self.data, dtype=np.int64)
            self.num_samples = 1

    def __len__(self):
        return max(1, self.num_samples)

    def __getitem__(self, idx):
        if self.total_tokens <= self.block_size + 1:
            # Short sample wrap/pad
            chunk = np.pad(self.data, (0, max(0, self.block_size + 1 - self.total_tokens)), mode='wrap')
            x = torch.tensor(chunk[:self.block_size], dtype=torch.long)
            y = torch.tensor(chunk[1:self.block_size + 1], dtype=torch.long)
            return x, y

        start_idx = idx * self.block_size
        chunk = self.data[start_idx : start_idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1].astype(np.int64), dtype=torch.long)
        y = torch.tensor(chunk[1:].astype(np.int64), dtype=torch.long)
        return x, y


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr: float, max_lr: float):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr / max_lr + (1.0 - min_lr / max_lr) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, val_loader, device, max_val_batches: int = 20) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_val_batches:
                break
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            total_loss += outputs["loss"].item()
            batches += 1
    
    avg_loss = total_loss / max(1, batches)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap to prevent math overflow
    model.train()
    return avg_loss, perplexity


def train(
    preset: str = "nano-25m",
    data_dir: str = "data/processed",
    output_dir: str = "checkpoints/pretrain",
    max_steps: int = 1000,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 6e-4,
    min_lr: float = 6e-5,
    warmup_steps: int = 50,
    save_interval: int = 250,
    eval_interval: int = 100,
    device: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Device Selection (MPS for M4 Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[+] Using Apple Silicon Metal (MPS) device for training.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("[+] Using NVIDIA CUDA GPU for training.")
        else:
            device = torch.device("cpu")
            print("[!] Using CPU for training.")
    else:
        device = torch.device(device)

    # 2. Config & Model Initialization
    config = CONFIG_PRESETS.get(preset, CONFIG_PRESETS["nano-25m"])
    
    # Check if custom tokenizer exists and sync vocab size
    tokenizer_json = os.path.join(data_dir, "..", "checkpoints", "tokenizer", "tokenizer.json")
    if not os.path.exists(tokenizer_json):
        tokenizer_json = "checkpoints/tokenizer/tokenizer.json"
    if os.path.exists(tokenizer_json):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_json)
        config.vocab_size = max(config.vocab_size, tok.get_vocab_size())

    config.save_pretrained(os.path.join(output_dir, "config.json"))

    print(f"[*] Initializing Tamil Nano LLM [{preset}]...")
    model = TamilNanoForCausalLM(config).to(device)
    param_info = model.count_parameters()
    print(f"[+] Total Parameters: {param_info['total']:,} ({param_info['total_millions']}M)")

    # 3. Datasets & Loaders
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_bin):
        raise FileNotFoundError(f"Missing {train_bin}. Please run preprocessing first.")

    train_dataset = MemoryMappedPretrainDataset(train_bin, block_size=config.max_position_embeddings)
    val_dataset = MemoryMappedPretrainDataset(val_bin, block_size=config.max_position_embeddings)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )
    scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr, learning_rate)

    # 5. Training Loop
    model.train()
    step = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"[*] Starting Pre-training: {max_steps} steps | Effective Batch: {batch_size * grad_accum_steps} | LR: {learning_rate}")
    
    train_iter = iter(train_loader)

    while step < max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            loss = outputs["loss"] / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        running_loss += accum_loss

        # Logging
        if step % 20 == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            tokens_per_sec = (step * batch_size * grad_accum_steps * config.max_position_embeddings) / (time.time() - start_time)
            print(f"Step [{step:4d}/{max_steps}] | Loss: {accum_loss:.4f} | LR: {current_lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s")

        # Evaluation
        if step % eval_interval == 0 or step == max_steps:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            print(f"--- Eval @ Step {step}: Val Loss = {val_loss:.4f} | Val Perplexity = {val_ppl:.2f} ---")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = os.path.join(output_dir, "best_model.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config.to_dict(),
                }, best_checkpoint)
                print(f"[✓] New best checkpoint saved to {best_checkpoint}")

        # Regular Checkpoint
        if step % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
            }, ckpt_path)

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, final_path)
    print(f"\n[✓] Pre-training completed successfully! Final model saved to {final_path}")


if __name__ == "__main__":
    from typing import Tuple
    parser = argparse.ArgumentParser(description="Pretrain Tamil Nano LLM")
    parser.add_argument("--preset", default="nano-25m", choices=["nano-25m", "micro-60m", "mini-125m"])
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="checkpoints/pretrain")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train(
        preset=args.preset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
    )
