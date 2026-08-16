#!/bin/bash
# ==============================================================================
# Tamil Nano LLM - End-to-End Training & Evaluation Pipeline
# Optimized for Apple Silicon M4 / 16GB RAM (MPS Accelerated)
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"

# Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================================="
echo "    TAMIL NANO LLM (MICRO-LLM) END-TO-END PIPELINE       "
echo "=========================================================="

# 1. Data Collection & Preparation
echo -e "\n[Step 1/5] Gathering and cleaning Tamil datasets..."
python3 src/data/download_corpus.py --output_dir data

# 2. Train Custom Tamil BPE Tokenizer
echo -e "\n[Step 2/5] Training Tamil-specialized BPE Tokenizer..."
python3 src/tokenizer/train_tokenizer.py \
    --data_files data/raw/pretrain_tamil.txt \
    --output_dir checkpoints/tokenizer \
    --vocab_size 8192 \
    --min_freq 1

# 3. Preprocess & Serialize into Binary Shards
echo -e "\n[Step 3/5] Preprocessing and creating binary token shards..."
python3 src/data/preprocess.py \
    --pretrain_file data/raw/pretrain_tamil.txt \
    --sft_file data/sft/instruct_tamil.jsonl \
    --tokenizer checkpoints/tokenizer/tokenizer.json \
    --out_dir data/processed

# 4. Pretrain from Scratch
echo -e "\n[Step 4/5] Pretraining Tamil Nano LLM (MPS / Metal Accelerated)..."
python3 src/train/train_pretrain.py \
    --preset nano-25m \
    --data_dir data/processed \
    --output_dir checkpoints/pretrain \
    --max_steps 300 \
    --batch_size 4 \
    --grad_accum 2 \
    --lr 6e-4

# 5. Supervised Fine-Tuning (SFT) for Chat, Translation & NLP
echo -e "\n[Step 5/5] Supervised Fine-Tuning (SFT) across multi-task instructions..."
python3 src/train/train_sft.py \
    --base_model checkpoints/pretrain/best_model.pt \
    --sft_data data/processed/sft_processed.jsonl \
    --output_dir checkpoints/sft \
    --epochs 5 \
    --batch_size 2 \
    --lr 3e-4

echo -e "\n=========================================================="
echo "    [✓] PIPELINE COMPLETED SUCCESSFULLY!                 "
echo "    To test interactively, run:                          "
echo "    python3 src/inference/generate.py                    "
echo "=========================================================="
