"""
Tamil Speech Recognition (ASR) Pipeline
Bridges Audio Voice Input to Tamil Text for the Tamil Nano LLM.
Supports Whisper (OpenAI / HuggingFace) and AI4Bharat IndicASR.
"""
import os
import argparse
from typing import Optional


class TamilSpeechPipeline:
    def __init__(self, model_id: str = "openai/whisper-small", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.pipe = None
        print(f"[*] Initializing Tamil ASR with model: {model_id}")

    def load_model(self):
        try:
            import torch
            from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

            torch_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype = torch.float16 if torch_device != "cpu" else torch.float32

            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(torch_device)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=torch_device,
            )
            print(f"[+] Loaded Whisper model on {torch_device}")
        except Exception as e:
            print(f"[!] Note: Whisper pipeline loading deferred or error: {e}")

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribes a Tamil speech audio file into Tamil text."""
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        if self.pipe is None:
            self.load_model()

        result = self.pipe(
            audio_file_path,
            generate_kwargs={"language": "tamil", "task": "transcribe"},
            return_timestamps=False,
        )
        text = result["text"].strip()
        return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tamil Speech Recognition (ASR)")
    parser.add_argument("--audio", type=str, help="Path to Tamil audio file (.wav/.mp3)")
    parser.add_argument("--model", type=str, default="openai/whisper-small")
    args = parser.parse_args()

    if args.audio:
        pipeline = TamilSpeechPipeline(model_id=args.model)
        transcription = pipeline.transcribe(args.audio)
        print(f"\n[Transcribed Tamil Text]: {transcription}")
    else:
        print("Tamil ASR module ready. Pass --audio <path_to_audio> to transcribe.")
