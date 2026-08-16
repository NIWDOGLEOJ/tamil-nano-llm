# Tamil Nano/Micro LLM (தமிழ் நுண்ணிய மாதிரி)

An end-to-end production framework and codebase for building, pretraining from scratch, instruction fine-tuning (SFT), evaluating, and deploying a Tamil-specialized Small Language Model on Apple Silicon M4 (16GB RAM) and Cloud GPUs.

---

## 🌟 Key Innovation Areas

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
   - Whisper ASR integration + Tamil Nano LLM + Text-to-Speech (TTS) audio synthesis.
6. **Edge & Mobile Export**:
   - SafeTensors, TorchScript JIT, and ONNX export for edge, iOS, macOS, and Android deployment.
7. **Interactive Web Application & REST API**:
   - FastAPI server with a modern Liquid Dark UI featuring dedicated tabs for all AI tools.

---

## 🏛️ Architecture Details

* **Model Type**: Autoregressive Decoder-Only Transformer
* **Positional Embeddings**: Rotary Positional Embedding (RoPE) ($\theta = 10,000$)
* **Attention Mechanism**: Grouped-Query Attention (GQA) with FlashAttention / MPS SDPA
* **Feed-Forward**: SwiGLU ($\text{SiLU}(W_{\text{gate}} x) \odot W_{\text{up}} x \cdot W_{\text{down}}$)
* **Normalization**: Root Mean Square Layer Normalization (RMSNorm)
* **Weight Sharing**: Tied input embeddings and output projection head
* **Context Window**: 512 tokens (extendable to 2,048)
* **Total Parameters**: 12.59 Million (`nano-25m` baseline)

---

## 📂 Project Structure

```
tamil-nano-llm/
├── checkpoints/              # Model weights, tokenizers, configs
│   ├── tokenizer/            # Tamil BPE tokenizer (8k vocab)
│   ├── pretrain/             # Pretrained weights (best_model.pt)
│   ├── sft/                  # Supervised fine-tuned weights (tamil_nano_instruct.pt)
│   └── exported/             # SafeTensors & TorchScript for Edge Deployment
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
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Launch the Interactive Web Portal
The Web UI provides a unified interface for Chat, Translation, Paraphrasing, NER, Voice Assistant, and Model Metrics:
```bash
./venv/bin/python3 -m uvicorn src.inference.web_demo:app --port 8000 --host 127.0.0.1
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

### 2. Interactive Terminal CLI
Test the model directly from your terminal:
```bash
# General Conversation
./venv/bin/python3 src/inference/generate.py --prompt "வணக்கம்! நீங்கள் யார்?"

# Machine Translation
./venv/bin/python3 src/inference/generate.py --prompt "Translate the following English sentence to Tamil:\nKnowledge is power."

# Interactive Chat Shell
./venv/bin/python3 src/inference/generate.py
```

---

### 3. Run Quality Evaluation & Benchmark Suite
Assess the model across Machine Translation, Paraphrasing, and Entity Extraction:
```bash
./venv/bin/python3 src/eval/benchmark.py
```

---

### 4. Voice AI Assistant (Speech-to-Speech)
```bash
./venv/bin/python3 src/speech/voice_chat.py --audio sample_tamil_speech.wav
```

---

### 5. Apple MLX QLoRA Fine-Tuning on M4 (Optional 1.5B/3B Scaling)
To fine-tune larger foundation models (e.g. Qwen 2.5 1.5B / Llama 3.2 1B) directly using M4 Unified Memory and Apple MLX:
```bash
# 1. Prepare MLX dataset
./venv/bin/python3 src/train/train_mlx_lora.py

# 2. Run MLX LoRA Fine-Tuning
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

### 6. Model Export for Edge & Mobile Devices
Export weights for iOS, macOS CoreML, and embedded devices:
```bash
./venv/bin/python3 src/quantization/export_model.py
```
Output files saved to `checkpoints/exported/`:
- `model.safetensors`
- `model_scripted.pt` (TorchScript JIT)
- `config.json`

---

## 📊 REST API Specifications

| Method | Endpoint | Payload | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/info` | None | Returns model metadata, parameter counts, and config |
| `POST` | `/api/generate` | `{"message": "...", "task": "chat"}` | General Tamil conversational response |
| `POST` | `/api/generate` | `{"message": "...", "task": "translation_en_ta"}` | English $\rightarrow$ Tamil translation |
| `POST` | `/api/generate` | `{"message": "...", "task": "translation_ta_en"}` | Tamil $\rightarrow$ English translation |
| `POST` | `/api/generate` | `{"message": "...", "task": "paraphrase"}` | Tamil sentence paraphrasing |
| `POST` | `/api/generate` | `{"message": "...", "task": "ner"}` | Named entity extraction |
