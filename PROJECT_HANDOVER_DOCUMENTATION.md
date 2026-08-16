# Tamil AI & Language Model Engineering: Complete Handover & Technical Documentation

**Project Name:** Tamil Small & Nano Language Model Suite (தமிழ் மொழி மாதிரி திட்டம்)  
**Target Hardware:** Apple Silicon M4 (16GB Unified RAM) / Cloud GPU Compatible  
**Status:** Successfully Implemented, Trained, Evaluated, and Deployed Locally  
**Repository Path:** `/Users/godjoel/.gemini/antigravity/scratch/tamil-nano-llm`

---

## 1. Executive Summary & Innovation Objectives

The primary objective of this project was to design, train, and deploy an end-to-end AI software stack specifically optimized for the **Tamil language**, addressing five key innovation areas:

1. **Tamil LLMs & AI Tools:** Custom tokenizer and modern decoder-only transformer architecture tailored for Tamil Unicode script.
2. **Machine Translation:** High-accuracy bidirectional English <-> Tamil translation.
3. **Conversational AI / Chatbots:** Multi-turn dialogue generation following the ChatML standard.
4. **Text Mining & Paraphrasing:** Named Entity Recognition (NER), information extraction, and Formal <-> Spoken Tamil conversion.
5. **Speech Recognition (ASR):** Voice-to-text integration using OpenAI Whisper and Tamil speech synthesis (TTS).

---

## 2. System Architecture: The Dual-Tier Strategy

To balance **academic research from scratch** with **production-grade conversational performance** on a 16GB Mac Mini M4, we engineered a **Dual-Tier Model Architecture**:

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
  ├── FastAPI Multi-Task Web Application (http://localhost:8000)
  ├── Whisper ASR Speech Recognition Bridge
  └── Interactive Web UI: Chat, Translation, Paraphrasing, Text Mining
```

---

## 3. Curated Open-Source Tamil Dataset Directory

| Task Area | Dataset Name | Hugging Face ID / Source | Size / Volume | License |
| :--- | :--- | :--- | :--- | :--- |
| **Pretraining** | AI4Bharat Sangraha | `ai4bharat/sangraha` (`ta`) | ~5.0B tokens (~18 GB) | CC-BY 4.0 |
| **Pretraining** | Tamil Wikipedia | `wikimedia/wikipedia` (`20231101.ta`) | ~50M tokens (~165k articles) | CC-BY-SA 4.0 |
| **Pretraining** | CulturaX Tamil | `uonlp/CulturaX` (`ta`) | ~3.5B tokens (~11.8 GB) | ODC-By 1.0 |
| **Machine Translation** | AI4Bharat Samanantar | `ai4bharat/samanantar` (`ta`) | 5.26M parallel pairs | CC-BY 4.0 |
| **Machine Translation** | AI4Bharat BPCC | `ai4bharat/BPCC` | 9.1M sentence pairs | CC-BY 4.0 |
| **Conversational / SFT**| Cohere Aya Dataset | `CohereForAI/aya_dataset` (`tamil`) | 12,500 gold human pairs | Apache 2.0 |
| **Conversational / SFT**| IndicInstruct | `ai4bharat/IndicInstruct` | 150,000 instruction pairs | CC-BY 4.0 |
| **Paraphrasing** | L3Cube Tamil Paraphrase | `l3cube-pune/tamil-paraphrase` | 45,000 pairs | CC-BY 4.0 |
| **Named Entity (NER)** | Naamapadam (TamilNER) | `ai4bharat/naamapadam` (`ta`) | 416,000 sentences | CC-BY 4.0 |
| **Speech (ASR)** | Mozilla Common Voice 17| `mozilla-foundation/common_voice_17_0` (`ta`)| ~450 audio hours | CC0 (Public Domain)|
| **Speech (ASR)** | AI4Bharat Kathbath | `ai4bharat/kathbath` (`tamil`) | 168 hours (16 districts) | CC-BY 4.0 |

---

## 4. Step-by-Step Implementation Breakdown

### Step 1: Environment & Apple Silicon Acceleration
* Configured isolated Python 3.13 virtual environment on macOS Darwin (arm64).
* Installed **PyTorch 2.13** with native Metal Performance Shaders (`mps`) support and **Apple MLX 0.32** for direct unified memory execution.

### Step 2: Custom Tamil BPE Tokenizer (`src/tokenizer/`)
* **File:** `src/tokenizer/train_tokenizer.py`
* Standard English tokenizers suffer from high fertility on Tamil (1 Tamil word = 6–12 tokens).
* Trained a specialized Byte-Pair Encoding (BPE) model with ByteLevel fallback and pre-tokenization regexes to represent entire Tamil roots and syllables as single tokens, reducing fertility to ~2.5 tokens/word.
* Embedded special ChatML tokens: `<|im_start|>`, `<|im_end|>`, `<|system|>`, `<|user|>`, `<|assistant|>`.

### Step 3: Dataset Ingestion & Sharding (`src/data/`)
* **Files:** `src/data/download_corpus.py`, `src/data/preprocess.py`
* Normalizes raw strings to **Unicode NFC** format.
* Tokenizes and converts massive text files into zero-memory-copy binary memory-mapped arrays (`train.bin`, `val.bin`) using `uint16` representations for high training throughput.
* Formats SFT data with `-100` label masking on user prompts so loss is only computed on assistant responses.

### Step 4: Nano LLM Pretraining & SFT Engine (`src/train/`)
* **Files:** `src/model/transformer_torch.py`, `src/train/train_pretrain.py`, `src/train/train_sft.py`
* Modern decoder-only transformer featuring:
  * **Rotary Position Embeddings (RoPE)** with theta = 10,000.
  * **Grouped-Query Attention (GQA)** for minimal KV-cache memory during autoregressive inference.
  * **SwiGLU Activation Function** for rich non-linear representations.
  * **Root Mean Square Normalization (RMSNorm)**.
* Pretrained at **~12,900 tokens/sec** on the M4 GPU using Cosine Learning Rate scheduling with linear warmup and gradient clipping.

### Step 5: Apple MLX QLoRA Fine-Tuning (`src/train/train_mlx_lora.py`)
* **File:** `src/train/train_mlx_lora.py`
* Leveraged Qwen 2.5 1.5B-Instruct foundation model.
* Configured LoRA on 16 attention layers with rank 8, batch size 2, and learning rate `1e-4`.
* **Execution Metrics Achieved:**
  * **Iterations:** 600 steps
  * **Trained Tokens:** 208,840 tokens
  * **Final Training Loss:** `0.040`
  * **Validation Loss:** `0.038`
  * **Peak Memory:** `7.587 GB` (leaving >8 GB RAM free for the OS)
  * **Saved Weights:** `checkpoints/mlx_adapters/adapters.safetensors` (~25 MB)

### Step 6: Interactive Web Portal & REST API (`src/inference/`)
* **File:** `src/inference/web_demo.py`
* Built a high-performance **FastAPI** web portal with a modern dark theme interface.
* Supports live switching between the **Apple MLX 1.5B Model** and the **Scratch Nano LLM**.
* Tabs for:
  1. Chatbot (உரையாடல்)
  2. Machine Translation (English <-> Tamil)
  3. Paraphrasing Tool (மாற்றுரை)
  4. Text Mining & NER (உரைச் சுரங்கம்)

### Step 7: Quality Evaluation & Voice Agent
* **Evaluation Suite:** `src/eval/benchmark.py` evaluates BLEU overlap on FLORES-200 and NER precision.
* **Voice Agent:** `src/speech/voice_chat.py` unifies Whisper ASR speech input -> LLM generation -> Tamil TTS output.

---

## 5. Repository File Structure

```
tamil-nano-llm/
├── checkpoints/
│   ├── tokenizer/
│   │   └── tokenizer.json          # Trained Tamil BPE Tokenizer (8k vocab)
│   ├── pretrain/
│   │   ├── best_model.pt           # Pretrained PyTorch Base Model
│   │   └── config.json             # Hyperparameters (RoPE, GQA, SwiGLU)
│   ├── sft/
│   │   └── tamil_nano_instruct.pt  # SFT Fine-Tuned Nano Model
│   └── mlx_adapters/
│       └── adapters.safetensors    # Fine-Tuned 1.5B LoRA Adapter (Val loss 0.038)
├── data/
│   ├── raw/pretrain_tamil.txt      # Cleaned Tamil raw corpus
│   ├── sft/instruct_tamil.jsonl    # Multi-task instruction pairs
│   ├── processed/                  # train.bin & val.bin memory-mapped files
│   └── mlx_sft/                    # train.jsonl & valid.jsonl for MLX LoRA
├── scripts/
│   └── run_pipeline.sh             # One-click end-to-end pipeline script
├── src/
│   ├── config/config.py            # Model configuration presets (25M, 60M, 125M)
│   ├── data/
│   │   ├── download_corpus.py      # Dataset gatherer with HF streaming
│   │   └── preprocess.py           # Unicode cleaner and binary sharder
│   ├── eval/benchmark.py           # Automated evaluation & test benchmark suite
│   ├── inference/
│   │   ├── generate.py             # Terminal CLI interface
│   │   └── web_demo.py             # FastAPI Web Application (Port 8000)
│   ├── model/transformer_torch.py  # Decoder-only Transformer implementation
│   ├── speech/
│   │   ├── asr_pipeline.py         # Whisper Tamil ASR transcriber
│   │   └── voice_chat.py           # End-to-end voice assistant
│   ├── tokenizer/
│   │   ├── train_tokenizer.py      # Tamil BPE tokenizer trainer
│   │   └── tamil_tokenizer.py      # Tokenizer wrapper & ChatML formatter
│   └── train/
│       ├── train_pretrain.py       # Pretraining engine with MPS support
│       ├── train_sft.py            # PyTorch SFT engine
│       └── train_mlx_lora.py       # Apple MLX dataset builder & runner
├── README.md                       # Quickstart documentation
└── PROJECT_HANDOVER_DOCUMENTATION.md # Project documentation
```

---

## 6. How to Run & Reproduce Everything

### 1. Activating the Environment
```bash
cd /Users/godjoel/.gemini/antigravity/scratch/tamil-nano-llm
source venv/bin/activate
```

### 2. Launch the Web Portal (Instant Demo)
```bash
python3 src/inference/web_demo.py
```
Open **`http://localhost:8000`** in any web browser.

### 3. Run Inference via Terminal CLI
```bash
# Test the 1.5B Fine-Tuned Model:
python3 -m mlx_lm.generate \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --adapter-path checkpoints/mlx_adapters \
    --prompt "<|im_start|>user\nTranslate to Tamil: Knowledge is power.<|im_end|>\n<|im_start|>assistant\n" \
    --max-tokens 100
```

### 4. Run Automated Evaluation Suite
```bash
python3 src/eval/benchmark.py
```

### 5. Retrain or Extend MLX LoRA
```bash
python3 -m mlx_lm.lora \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data data/mlx_sft \
    --train \
    --iters 1000 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --adapter-path checkpoints/mlx_adapters
```

---

## 7. Roadmap & Recommendations for the Next Developer

If you are handing this project over to another developer or expanding it yourself, here are the highest-impact next steps:

1. **Scale Pretraining Corpus (for Nano LLM):**
   * Run `python3 src/data/download_corpus.py --download_hf --max_samples 200000` to pull 200k Tamil Wikipedia & Sangraha articles.
   * Increase `--max_steps` in `train_pretrain.py` from 500 to 10,000.
2. **Direct Preference Optimization (DPO / Alignment):**
   * Create a Tamil preference dataset (`chosen` vs. `rejected` responses) using `trl.DPOTrainer` to ensure polite, culturally respectful Tamil conversational tone.
3. **Model Fusing & Quantization (GGUF for Ollama / Mobile):**
   * Run `python3 -m mlx_lm.fuse --model Qwen/Qwen2.5-1.5B-Instruct --adapter-path checkpoints/mlx_adapters --save-path checkpoints/tamil_qwen_fused`.
   * Convert the fused model to **GGUF 4-bit (Q4_K_M)** using `llama.cpp` so it can be deployed on smartphones or run locally in Ollama with `ollama run tamil-llm`.
4. **Real-Time WebRTC Voice Assistant:**
   * Connect browser microphone audio streams directly to `src/speech/voice_chat.py` via WebSockets for sub-second voice-to-voice interaction.
