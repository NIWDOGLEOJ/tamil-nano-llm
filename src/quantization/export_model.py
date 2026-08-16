"""
Model Export & Edge Deployment Engine
Exports trained Tamil Nano LLM to SafeTensors, TorchScript, and ONNX formats for edge, iOS, macOS, and mobile deployment.
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.config import TamilNanoConfig
from src.model.transformer_torch import TamilNanoForCausalLM


class JITExportWrapper(nn.Module):
    """Wrapper to produce plain logits tensor without dictionary for clean TorchScript / ONNX tracing."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        res = self.model(input_ids, return_dict=True)
        return res["logits"]


def export_to_safetensors(model, config, output_dir: str):
    try:
        from safetensors.torch import save_file
        save_path = os.path.join(output_dir, "model.safetensors")
        state_dict = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, save_path)
        print(f"[✓] Exported SafeTensors weights to {save_path} ({os.path.getsize(save_path) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"[!] SafeTensors export notice: {e}")


def export_to_torchscript(model, config, output_dir: str):
    save_path = os.path.join(output_dir, "model_scripted.pt")
    wrapper = JITExportWrapper(model)
    wrapper.eval()
    dummy_input = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)
    
    try:
        traced_model = torch.jit.trace(wrapper, dummy_input)
        traced_model.save(save_path)
        print(f"[✓] Exported TorchScript model to {save_path} ({os.path.getsize(save_path) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"[!] TorchScript trace note: {e}")


def export_to_onnx(model, config, output_dir: str):
    save_path = os.path.join(output_dir, "model.onnx")
    wrapper = JITExportWrapper(model)
    wrapper.eval()
    dummy_input = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)

    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            save_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17
        )
        print(f"[✓] Exported ONNX model to {save_path} ({os.path.getsize(save_path) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"[!] ONNX export note: {e}")


def export_all(model_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Loading model from {model_path} for edge export...")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("config", {})
    config = TamilNanoConfig.from_dict(config_dict)
    
    model = TamilNanoForCausalLM(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 1. Save Config
    config.save_pretrained(os.path.join(output_dir, "config.json"))

    # 2. Export SafeTensors
    export_to_safetensors(model, config, output_dir)

    # 3. Export TorchScript
    export_to_torchscript(model, config, output_dir)

    # 4. Export ONNX
    export_to_onnx(model, config, output_dir)

    print(f"\n[✓] Export complete! All deployment artifacts saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Tamil Nano LLM for edge deployment")
    parser.add_argument("--model", default="checkpoints/sft/tamil_nano_instruct.pt")
    parser.add_argument("--output_dir", default="checkpoints/exported")
    args = parser.parse_args()

    if os.path.exists(args.model):
        export_all(args.model, args.output_dir)
    else:
        print(f"[!] Model {args.model} not found.")
