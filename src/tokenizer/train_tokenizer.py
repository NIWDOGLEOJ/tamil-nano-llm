"""
Custom Tamil Byte-Pair Encoding (BPE) Tokenizer Trainer
Designed to drastically reduce token fertility for Tamil script while maintaining English & Code support.
"""
import os
import argparse
from typing import List, Optional
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    processors,
    Regex
)


SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|translate_en_ta|>",
    "<|translate_ta_en|>",
    "<|paraphrase|>",
    "<|extract_ner|>",
]


def train_bpe_tokenizer(
    input_files: List[str],
    output_dir: str,
    vocab_size: int = 16384,
    min_frequency: int = 2,
) -> Tokenizer:
    """Trains a BPE tokenizer specifically tailored for Tamil Unicode script."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Training Tamil BPE Tokenizer with vocab size {vocab_size} on files: {input_files}")

    # Initialize BPE Model with unknown token replacement
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

    # Pre-tokenization: Split by whitespace, digits, punctuation, and byte-level fallback
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    ])

    # Decoder: ByteLevel decoder to seamlessly reconstruct UTF-8 Tamil characters
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train on corpus files
    tokenizer.train(input_files, trainer)

    # Post-processor: Add BOS / EOS handling
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    # Save tokenizer files
    save_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"[+] Tokenizer saved successfully to {save_path}")

    # Print validation stats
    test_sentences = [
        "வணக்கம்! இது தமிழ் மொழி சார்ந்த நுண்ணிய மாதிரி (Nano LLM).",
        "Machine translation from English to Tamil is working efficiently.",
        "செயற்கை நுண்ணறிவு தொழில்நுட்பம் தமிழ் மொழியில் மிக வேகமாக வளர்ந்து வருகிறது."
    ]
    print("\n--- Tokenizer Validation Test ---")
    for text in test_sentences:
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        decoded = tokenizer.decode(encoding.ids)
        print(f"Original : {text}")
        print(f"Tokens   : {len(tokens)} tokens -> {tokens[:10]}...")
        print(f"Decoded  : {decoded}")
        print(f"Fertility: {len(tokens) / len(text.split()):.2f} tokens/word\n")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tamil BPE Tokenizer")
    parser.add_argument("--data_files", nargs="+", required=True, help="Path to raw Tamil text files")
    parser.add_argument("--output_dir", default="checkpoints/tokenizer", help="Directory to save tokenizer")
    parser.add_argument("--vocab_size", type=int, default=16384, help="Vocabulary size")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum token frequency")
    args = parser.parse_args()

    train_bpe_tokenizer(
        input_files=args.data_files,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )
