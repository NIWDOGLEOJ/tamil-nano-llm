# Tamil Small & Nano Language Model (SLM) Suite 🚀
### தமிழ் நுண்ணிய & சிறிய மொழி மாதிரி திட்டம்

[![GitHub Release](https://img.shields.io/github/v/release/NIWDOGLEOJ/tamil-nano-llm?color=orange&label=Release)](https://github.com/NIWDOGLEOJ/tamil-nano-llm/releases/tag/v1.0.0)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M4%20MPS%20%26%20MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![GGUF Quantized](https://img.shields.io/badge/Format-GGUF%20(Q8__0)-green)](https://github.com/ggerganov/llama.cpp)
[![FastAPI Web Portal](https://img.shields.io/badge/Web%20UI-FastAPI%20%26%20REST%20API-009688?logo=fastapi)](http://localhost:8000)

An end-to-end production framework and codebase for building, pretraining from scratch, instruction fine-tuning (SFT), GGUF quantizing, evaluating, and deploying a Tamil-specialized Language Model on **Apple Silicon M4 (16GB RAM)**, Cloud GPUs, and Edge/Mobile devices.

---

## 🌟 Key Innovation Areas Covered

1. **Tamil LLMs & AI Architecture**:
   - Custom Byte-Pair Encoding (BPE) tokenizer (8,192 vocab) with low fertility ratio on Tamil script.
   - Modern Decoder-Only Transformer (Rotary Positional Embeddings, SwiGLU, RMSNorm, Grouped-Query Attention).
   - PyTorch Metal Performance Shaders (MPS) acceleration + Apple MLX QLoRA support.
2. **Machine Translation**:
   - High-accuracy bidirectional translation between English $\leftrightarrow$ Tamil.
3. **Conversational AI / Chatbot**:
   - ChatML dialogue formatting (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`).
4. **Text Mining & Paraphrasing**:
   - Named Entity Recognition (NER: Persons, Places, Organizations) and formal $\leftrightarrow$ spoken Tamil style transfer.
5. **Voice AI Assistant**:
   - Whisper ASR integration + Tamil LLM + Text-to-Speech (TTS) audio synthesis.
6. **GGUF Quantization & Ollama Deployment**:
   - High-speed 8-bit quantized GGUF format for one-command execution in Ollama and llama.cpp.
7. **Interactive Web Application & REST API**:
   - FastAPI server with a modern Dark Theme UI featuring dedicated tabs for all AI tools.

---

## 🏛️ Dual-Tier System Architecture

```
[ Data Layer ]
  ├── Raw Tamil Corpora (Sangraha, Wikipedia, OSCAR)
  ├── Instruction Data (Aya, Samanantar, Naamapadam)
  └── Custom Tamil BPE Tokenizer (8k Vocab)

[ Tier 1: Custom Nano LLM From Scratch ]
  ├── Parameters: ~25M (Tied embeddings)
  ├── PyTorch with Apple Silicon Metal (MPS) Acceleration
  ├── Speed: ~12,900 tokens/sec on M4 GPU
  └── Purpose: Lightweight, edge-deployable, educational from-scratch baseline

[ Tier 2: Apple MLX 1.5B Model Fine-Tuning ]
  ├── Base Model: Qwen 2.5 1.5B-Instruct
  ├── Technique: Apple MLX QLoRA (600 iterations on Unified Memory)
  ├── Result: Train Loss 0.040, Val Loss 0.038, Peak RAM: 7.58 GB
  └── Purpose: Production-grade chatbot, high-accuracy translation, complex reasoning

[ Serving & Interface Layer ]
  ├── GGUF 8-bit Model (checkpoints/tamil_qwen_1.5b_instruct_q8_0.gguf)
  ├── Ollama Integration (Modelfile)
  ├── FastAPI Multi-Task Web Application (http://localhost:8000)
  ├── Whisper ASR Speech Recognition Bridge
  └── Interactive Web UI: Chat, Translation, Paraphrasing, Text Mining
```

---

## 📂 Project Structure

```
tamil-nano-llm/
├── checkpoints/              # Model weights, tokenizers, configs
│   ├── tokenizer/            # Tamil BPE tokenizer (8k vocab)
│   ├── pretrain/             # Pretrained weights (best_model.pt)
│   ├── sft/                  # Supervised fine-tuned weights (tamil_nano_instruct.pt)
│   ├── mlx_adapters/         # Fine-tuned 1.5B LoRA Adapter (Val loss 0.038)
│   ├── exported/             # SafeTensors & TorchScript for Edge Deployment
│   └── tamil_qwen_1.5b_instruct_q8_0.gguf # Exported 8-bit GGUF (1.64 GB)
├── data/
│   ├── raw/                  # Cleaned raw Tamil text corpora
│   ├── sft/                  # Multi-task instruction JSONL files
│   ├── mlx_sft/              # Formatted train/val pairs for Apple MLX QLoRA
│   └── processed/            # Memory-mapped binary token shards (.bin)
├── scripts/
│   └── run_pipeline.sh       # Automated pipeline execution script
├── src/
│   ├── eval/
│   │   └── benchmark.py      # Multi-task evaluation & scoring suite
│   ├── model/
│   │   ├── config.py         # TamilNanoConfig & presets
│   │   └── transformer_torch.py # PyTorch Transformer implementation
│   ├── data/
│   │   ├── download_corpus.py# Dataset collection & multi-task generator
│   │   └── preprocess.py     # Binary serialization & ChatML packaging
│   ├── inference/
│   │   ├── generate.py       # Interactive CLI & generation pipeline
│   │   └── web_demo.py       # FastAPI Web Portal & REST API
│   ├── quantization/
│   │   ├── convert_hf_to_gguf.py # GGUF converter script
│   │   └── export_model.py   # SafeTensors, TorchScript & ONNX export
│   ├── speech/
│   │   ├── asr_pipeline.py   # Whisper Tamil ASR speech bridge
│   │   └── voice_chat.py     # End-to-end Voice AI Agent
│   ├── tokenizer/
│   │   ├── train_tokenizer.py# Tamil BPE Tokenizer training
│   │   └── tamil_tokenizer.py# Tokenizer wrapper & ChatML formatter
│   └── train/
│       ├── train_pretrain.py # Pretraining engine with Cosine Warmup
│       ├── train_sft.py      # Multi-task SFT trainer
│       └── train_mlx_lora.py # Apple MLX QLoRA fine-tuning for M4
├── Modelfile                 # Ollama model definition
├── PROJECT_HANDOVER_DOCUMENTATION.md # Comprehensive Handover & Architecture Docs
└── README.md
```

---

## 🚀 Quickstart & Execution

### 1. Launch the Interactive Web Portal
The Web UI provides a unified interface for Chat, Translation, Paraphrasing, NER, Voice Assistant, and Model Metrics:
```bash
source venv/bin/activate
python3 src/inference/web_demo.py
```
Open **`http://localhost:8000`** in your browser.

---

### 2. Run with Ollama (One Command)
```bash
# 1. Create the model in Ollama
ollama create tamil-llm -f Modelfile

# 2. Run and chat
ollama run tamil-llm
```

---

### 3. Interactive Terminal CLI
Test the model directly from your terminal:
```bash
# Test the 1.5B Fine-Tuned Model via MLX:
python3 -m mlx_lm.generate \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path checkpoints/mlx_adapters \
    --prompt "<|im_start|>user\nTranslate to Tamil: Artificial intelligence is transforming education.<|im_end|>\n<|im_start|>assistant\n" \
    --max-tokens 100
```

---

### 4. Run Quality Evaluation & Benchmark Suite
Assess the model across Machine Translation, Paraphrasing, and Entity Extraction:
```bash
python3 src/eval/benchmark.py
```

---

### 5. Voice AI Assistant (Speech-to-Speech)
```bash
python3 src/speech/voice_chat.py --audio sample_tamil_speech.wav
```

---

### 6. Retrain or Extend MLX LoRA Fine-Tuning
```bash
# 1. Prepare MLX dataset
python3 src/train/train_mlx_lora.py

# 2. Run MLX LoRA Fine-Tuning (M4 GPU Accelerated)
python3 -m mlx_lm.lora \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data data/mlx_sft \
    --train \
    --iters 600 \
    --batch-size 2 \
    --lora-layers 16 \
    --learning-rate 1e-4 \
    --adapter-path checkpoints/mlx_adapters
```

---

## 📊 REST API Specifications

| Method | Endpoint | Payload | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/info` | None | Returns active model metadata and acceleration backend |
| `POST` | `/api/generate` | `{"message": "...", "task": "chat"}` | General Tamil conversational response |
| `POST` | `/api/generate` | `{"message": "...", "task": "translation_en_ta"}` | English $\rightarrow$ Tamil translation |
| `POST` | `/api/generate` | `{"message": "...", "task": "translation_ta_en"}` | Tamil $\rightarrow$ English translation |
| `POST` | `/api/generate` | `{"message": "...", "task": "paraphrase"}` | Tamil sentence paraphrasing |
| `POST` | `/api/generate` | `{"message": "...", "task": "ner"}` | Named entity extraction |

---

## 📚 Complete Project Handover Documentation
For the full technical design, mathematical formulations, training logs, and collaborator roadmap, refer to **[`PROJECT_HANDOVER_DOCUMENTATION.md`](file:///Users/godjoel/.gemini/antigravity/scratch/tamil-nano-llm/PROJECT_HANDOVER_DOCUMENTATION.md)**.
