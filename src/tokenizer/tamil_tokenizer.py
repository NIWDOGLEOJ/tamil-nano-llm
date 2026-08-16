"""
Tamil Tokenizer Wrapper
Provides encoding, decoding, padding, batch tokenization, and ChatML templating.
"""
import os
import json
from typing import List, Dict, Union, Optional
from tokenizers import Tokenizer


class TamilTokenizer:
    def __init__(self, tokenizer_path: str):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Token IDs
        self.pad_token = "<|pad|>"
        self.bos_token = "<|bos|>"
        self.eos_token = "<|eos|>"
        self.unk_token = "<|unk|>"
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token) or 0
        self.bos_token_id = self.tokenizer.token_to_id(self.bos_token) or 1
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token) or 2
        self.unk_token_id = self.tokenizer.token_to_id(self.unk_token) or 3

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        text = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return text

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Formats messages into ChatML prompt format:
        <|im_start|>system
        You are a helpful Tamil AI assistant.<|im_end|>
        <|im_start|>user
        வணக்கம்!<|im_end|>
        <|im_start|>assistant
        வணக்கம்! நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?<|im_end|>
        """
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"
        
        return prompt

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 1024,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[List[int]]]:
        all_ids = []
        all_masks = []

        for text in texts:
            ids = self.encode(text, add_special_tokens=True)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            
            mask = [1] * len(ids)
            
            if padding and len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
                
            all_ids.append(ids)
            all_masks.append(mask)

        return {"input_ids": all_ids, "attention_mask": all_masks}
