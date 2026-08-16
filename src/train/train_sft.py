"""
Supervised Fine-Tuning (SFT) Engine for Tamil Multi-Task Instruction Following
Specializes the model for Chatbots, Machine Translation, Paraphrasing, and Text Mining.
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.config import TamilNanoConfig
from src.model.transformer_torch import TamilNanoForCausalLM


class SFTDataset(Dataset):
    def __init__(self, jsonl_file: str, max_length: int = 1024, pad_token_id: int = 0):
        self.samples = []
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        if not self.samples:
            # Fallback dummy sample
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.full((self.max_length,), -100, dtype=torch.long)
            }
        item = self.samples[idx % len(self.samples)]
        input_ids = item["input_ids"][:self.max_length]
        labels = item["labels"][:self.max_length]

        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "task": item.get("task", "general")
        }


def train_sft(
    base_model_path: str,
    sft_data_path: str,
    output_dir: str = "checkpoints/sft",
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    device: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    print(f"[*] Loading base model from {base_model_path} onto {device}...")
    
    # Load config and weights
    if os.path.exists(base_model_path):
        checkpoint = torch.load(base_model_path, map_location="cpu", weights_only=False)
        config_dict = checkpoint.get("config", {})
        config = TamilNanoConfig.from_dict(config_dict)
        model = TamilNanoForCausalLM(config)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        print(f"[!] Warning: Base model {base_model_path} not found. Initializing fresh config.")
        config = TamilNanoConfig()
        model = TamilNanoForCausalLM(config)

    model.to(device)
    model.train()

    dataset = SFTDataset(sft_data_path, max_length=min(config.max_position_embeddings, 512), pad_token_id=config.pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    print(f"[*] Starting SFT across {len(dataset)} samples for {epochs} epochs...")
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch [{epoch+1}/{epochs}] | Step {step} | Loss: {avg_loss:.4f}")

    # Save SFT Checkpoint
    sft_model_path = os.path.join(output_dir, "tamil_nano_instruct.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, sft_model_path)
    config.save_pretrained(os.path.join(output_dir, "config.json"))

    print(f"[✓] SFT Training complete! Saved final instruct model to {sft_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Tamil Nano LLM for SFT")
    parser.add_argument("--base_model", default="checkpoints/pretrain/best_model.pt")
    parser.add_argument("--sft_data", default="data/processed/sft_processed.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_sft(
        base_model_path=args.base_model,
        sft_data_path=args.sft_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
