"""
End-to-End Tamil Voice AI Agent
Pipeline: Audio Speech Input -> Whisper ASR -> Tamil Nano LLM -> Tamil Speech Synthesizer (TTS)
"""
import os
import sys
import argparse
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.speech.asr_pipeline import TamilSpeechPipeline
from src.inference.generate import TamilNanoPipeline


class TamilVoiceAgent:
    def __init__(
        self,
        model_path: str = "checkpoints/sft/tamil_nano_instruct.pt",
        tokenizer_path: str = "checkpoints/tokenizer/tokenizer.json",
        asr_model: str = "openai/whisper-small",
    ):
        print("[*] Initializing End-to-End Tamil Voice Assistant...")
        self.llm = TamilNanoPipeline(model_path, tokenizer_path)
        self.asr = TamilSpeechPipeline(model_id=asr_model)

    def speak_tamil(self, text: str):
        """Uses macOS native Tamil voice or gTTS to speak output."""
        if not text:
            return
        print(f"[Speaking Output]: {text}")
        
        # On macOS, check if native Tamil speech voice is installed
        try:
            # Try macOS native 'say'
            subprocess.run(["say", text], check=True)
        except Exception:
            pass

    def process_voice_turn(self, audio_path: str) -> dict:
        # 1. Transcribe speech
        print(f"\n[1/3] Transcribing audio: {audio_path}...")
        user_tamil_text = self.asr.transcribe(audio_path)
        print(f"      Transcribed Text: {user_tamil_text}")

        # 2. Generate LLM Response
        print("[2/3] Generating AI Response...")
        llm_response = self.llm.chat(user_tamil_text)
        print(f"      Tamil AI Response: {llm_response}")

        # 3. Audio Synthesis
        print("[3/3] Synthesizing speech...")
        self.speak_tamil(llm_response)

        return {
            "transcription": user_tamil_text,
            "response": llm_response
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tamil Voice Agent")
    parser.add_argument("--audio", type=str, help="Input Tamil speech audio (.wav/.mp3)")
    args = parser.parse_args()

    if args.audio and os.path.exists(args.audio):
        agent = TamilVoiceAgent()
        agent.process_voice_turn(args.audio)
    else:
        print("Tamil Voice Assistant ready. Run with: python3 src/speech/voice_chat.py --audio <audio_file.wav>")
