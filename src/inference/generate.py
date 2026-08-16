"""
Tamil Nano LLM Inference & Interactive CLI
Supports text completion, ChatML dialogue, translation, and task-based prompting.
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.config import TamilNanoConfig
from src.model.transformer_torch import TamilNanoForCausalLM
from src.tokenizer.tamil_tokenizer import TamilTokenizer


class TamilNanoPipeline:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"[*] Loading Tamil Nano LLM from {model_path} onto {self.device}...")
        self.tokenizer = TamilTokenizer(tokenizer_path)
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        config_dict = checkpoint.get("config", {})
        self.config = TamilNanoConfig.from_dict(config_dict)
        
        self.model = TamilNanoForCausalLM(self.config)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_tensor = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_tensor[0].tolist()[len(input_ids):]
        # Stop at eos or im_end
        im_end_id = self.tokenizer.tokenizer.token_to_id("<|im_end|>")
        if im_end_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(im_end_id)]

        return self.tokenizer.decode(generated_ids).strip()

    def chat(self, user_message: str, history: list = None) -> str:
        messages = history or []
        messages.append({"role": "user", "content": user_message})
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        response = self.generate(prompt)
        return response


def run_interactive_cli(pipeline: TamilNanoPipeline):
    print("\n" + "="*50)
    print("  TAMIL NANO LLM - INTERACTIVE CHAT & TOOLS")
    print("  Type 'exit' to quit. Prefix commands:")
    print("  - /translate <text>  (English to Tamil)")
    print("  - /paraphrase <text> (Rephrase Tamil text)")
    print("  - /ner <text>        (Extract Entities)")
    print("="*50 + "\n")

    history = []
    while True:
        try:
            user_input = input("\n[User]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting. நன்றி!")
                break

            if user_input.startswith("/translate "):
                text = user_input[len("/translate "):]
                prompt = f"<|im_start|>user\nTranslate the following English sentence to Tamil:\n{text}<|im_end|>\n<|im_start|>assistant\n"
                print(f"[Translation]: {pipeline.generate(prompt)}")
            elif user_input.startswith("/paraphrase "):
                text = user_input[len("/paraphrase "):]
                prompt = f"<|im_start|>user\nஇவ்வாக்கியத்தை வேறு வடிவில் மாற்றியமைத்து எழுதுக (Paraphrase):\n{text}<|im_end|>\n<|im_start|>assistant\n"
                print(f"[Paraphrase]: {pipeline.generate(prompt)}")
            elif user_input.startswith("/ner "):
                text = user_input[len("/ner "):]
                prompt = f"<|im_start|>user\nகீழ்க்கண்ட உரையிலிருந்து நபர்கள், இடங்கள் மற்றும் அமைப்புகளைப் பிரித்தெடுக்கவும்:\n{text}<|im_end|>\n<|im_start|>assistant\n"
                print(f"[Entities]: {pipeline.generate(prompt)}")
            else:
                response = pipeline.chat(user_input, history)
                print(f"[Tamil Nano LLM]: {response}")
                history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting. நன்றி!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference CLI for Tamil Nano LLM")
    parser.add_argument("--model", default="checkpoints/sft/tamil_nano_instruct.pt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer/tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    if os.path.exists(args.model) and os.path.exists(args.tokenizer):
        pipe = TamilNanoPipeline(args.model, args.tokenizer)
        if args.prompt:
            if "<|im_start|>" not in args.prompt:
                print(f"[Output]: {pipe.chat(args.prompt)}")
            else:
                print(f"[Output]: {pipe.generate(args.prompt)}")
        else:
            run_interactive_cli(pipe)
    else:
        print(f"[!] Model or Tokenizer not found at {args.model}, {args.tokenizer}.")
