"""
Interactive Web Application & REST API for Tamil Nano LLM
Features:
1. 💬 Conversational Chatbot (உரையாடல்)
2. 🌐 Bidirectional Translation En <-> Ta (மொழியாக்கம்)
3. 🔄 Paraphrase & Style Transfer (மாற்றுரை)
4. 🔍 Text Mining & Entity Recognition (உரைச் சுரங்கம்)
5. 🎙️ Voice AI Agent (குரல் உதவியாளர்)
6. 📊 Architecture & Benchmark Metrics (மாதிரி விவரங்கள்)
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.generate import TamilNanoPipeline

app = FastAPI(
    title="Tamil Nano LLM Portal",
    description="Multi-task Tamil AI Engine on Apple Silicon M4",
    version="1.0.0"
)

pipeline: Optional[TamilNanoPipeline] = None
mlx_model = None
mlx_tokenizer = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        model_path = "checkpoints/sft/tamil_nano_instruct.pt"
        if not os.path.exists(model_path):
            model_path = "checkpoints/pretrain/best_model.pt"
        tokenizer_path = "checkpoints/tokenizer/tokenizer.json"
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            try:
                pipeline = TamilNanoPipeline(model_path, tokenizer_path)
            except Exception as e:
                print(f"[!] Nano LLM load note: {e}")
    return pipeline


def get_mlx_model():
    global mlx_model, mlx_tokenizer
    adapter_path = "checkpoints/mlx_adapters"
    if mlx_model is None and os.path.exists(adapter_path):
        try:
            from mlx_lm import load
            print(f"[*] Loading Apple MLX LoRA Tamil Model from {adapter_path}...")
            mlx_model, mlx_tokenizer = load("Qwen/Qwen2.5-1.5B-Instruct", adapter_path=adapter_path)
            print("[✓] Loaded MLX LoRA 1.5B Tamil Model successfully on M4 Unified Memory!")
        except Exception as e:
            print(f"[!] MLX model loading note: {e}")
    return mlx_model, mlx_tokenizer


class ChatRequest(BaseModel):
    message: str
    task: Optional[str] = "chat"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150
    engine: Optional[str] = "auto"  # 'auto', 'mlx_1.5b', or 'nano_scratch'


@app.on_event("startup")
def startup_event():
    get_mlx_model()
    get_pipeline()


@app.get("/api/info")
def api_info():
    mlx_m, _ = get_mlx_model()
    has_mlx = mlx_m is not None
    return {
        "model_name": "Tamil AI Suite (Apple MLX 1.5B & Nano LLM)",
        "primary_engine": "Apple MLX LoRA (Qwen 2.5 1.5B)" if has_mlx else "Tamil Nano LLM",
        "mlx_available": has_mlx,
        "hardware_acceleration": "Apple Silicon M4 GPU & Neural Engine",
        "tasks": ["Conversational Chat", "Machine Translation", "Paraphrasing", "Named Entity Recognition", "Voice AI Assistant"]
    }


@app.post("/api/generate")
async def api_generate(req: ChatRequest):
    task = req.task
    text = req.message.strip()

    if task == "translation_en_ta":
        prompt = f"<|im_start|>user\nTranslate the following English sentence to Tamil:\n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif task == "translation_ta_en":
        prompt = f"<|im_start|>user\nTranslate this Tamil sentence to English:\n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif task == "paraphrase":
        prompt = f"<|im_start|>user\nஇவ்வாக்கியத்தை வேறு வடிவில் மாற்றியமைத்து எழுதுக (Paraphrase):\n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif task == "ner":
        prompt = f"<|im_start|>user\nகீழ்க்கண்ட உரையிலிருந்து நபர்கள், இடங்கள் மற்றும் அமைப்புகளைப் பிரித்தெடுக்கவும் (Named Entity Recognition):\n{text}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

    # 1. Try MLX LoRA Model first (Highest Quality)
    mlx_m, mlx_tok = get_mlx_model()
    if mlx_m is not None and req.engine != "nano_scratch":
        try:
            from mlx_lm import generate
            output = generate(
                mlx_m,
                mlx_tok,
                prompt=prompt,
                max_tokens=req.max_tokens,
                verbose=False
            ).strip()
            
            # Clean up assistant token tags if present
            if "<|im_end|>" in output:
                output = output.split("<|im_end|>")[0].strip()
            return {"status": "success", "engine": "mlx_1.5b", "task": task, "input": text, "response": output}
        except Exception as e:
            print(f"[!] MLX Generation error: {e}")

    # 2. Fallback to Nano LLM
    pipe = get_pipeline()
    if pipe is not None:
        output = pipe.generate(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature)
        return {"status": "success", "engine": "nano_scratch", "task": task, "input": text, "response": output}

    return JSONResponse({"status": "error", "message": "No model loaded. Please check checkpoint directories."})


@app.get("/", response_class=HTMLResponse)
async def serve_portal():
    html = """<!DOCTYPE html>
<html lang="ta">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tamil Nano LLM | தமிழ் நுண்ணிய மொழி மாதிரி</title>
    <style>
        :root {
            --primary: #ea580c;
            --primary-hover: #f97316;
            --bg: #0b0f17;
            --card-bg: #131b2a;
            --card-sub: #1e293b;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
            --border: #243046;
            --accent: #38bdf8;
            --success: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans Tamil', sans-serif; }
        body { background-color: var(--bg); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
        
        .header { text-align: center; margin-bottom: 1.75rem; max-width: 850px; }
        .header h1 { font-size: 2.25rem; color: #ffedd5; margin-bottom: 0.5rem; letter-spacing: -0.5px; font-weight: 800; }
        .header p { color: var(--text-sub); font-size: 1.05rem; line-height: 1.5; }
        
        .badge-row { display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap; margin-top: 0.75rem; }
        .badge { background: #321908; color: #fdba74; border: 1px solid #7c2d12; padding: 0.3rem 0.85rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
        .badge.accent { background: #0c2d48; color: #7dd3fc; border-color: #0369a1; }
        .status-dot { height: 8px; width: 8px; background-color: var(--success); border-radius: 50%; display: inline-block; }

        .container { width: 100%; max-width: 900px; background: var(--card-bg); border-radius: 16px; border: 1px solid var(--border); box-shadow: 0 15px 35px -5px rgba(0,0,0,0.6); overflow: hidden; }
        .tabs { display: flex; border-bottom: 1px solid var(--border); background: #0e1626; overflow-x: auto; scrollbar-width: none; }
        .tab-btn { flex: 1; min-width: 140px; padding: 1.1rem 0.75rem; background: none; border: none; color: var(--text-sub); font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: all 0.2s ease; text-align: center; display: flex; align-items: center; justify-content: center; gap: 6px; }
        .tab-btn:hover { color: var(--text-main); background: #172235; }
        .tab-btn.active { color: #fed7aa; border-bottom: 3px solid var(--primary); background: var(--card-bg); }

        .tab-content { padding: 2rem; display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.2s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

        .form-group { margin-bottom: 1.25rem; }
        label { display: block; font-size: 0.9rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.5rem; }
        textarea, input[type="text"], select { width: 100%; background: #0b111c; border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem; color: #f8fafc; font-size: 1rem; line-height: 1.5; resize: vertical; transition: border-color 0.2s; }
        textarea:focus, input[type="text"]:focus, select:focus { outline: none; border-color: var(--accent); }

        .quick-samples { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
        .sample-pill { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 0.25rem 0.65rem; border-radius: 6px; font-size: 0.8rem; cursor: pointer; transition: all 0.15s ease; }
        .sample-pill:hover { background: #334155; color: #fff; border-color: #475569; }

        .btn { background: var(--primary); color: #fff; border: none; padding: 0.85rem 1.6rem; border-radius: 10px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: background 0.2s ease, transform 0.1s ease; display: inline-flex; align-items: center; justify-content: center; gap: 0.5rem; }
        .btn:hover { background: var(--primary-hover); transform: translateY(-1px); }
        .btn:active { transform: translateY(0); }
        .btn.secondary { background: #1e293b; border: 1px solid #334155; color: #e2e8f0; }
        .btn.secondary:hover { background: #334155; }

        .result-box { margin-top: 1.5rem; background: #0b111c; border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; }
        .result-box h4 { font-size: 0.85rem; text-transform: uppercase; color: var(--accent); margin-bottom: 0.6rem; letter-spacing: 0.5px; display: flex; align-items: center; justify-content: space-between; }
        .result-text { font-size: 1.05rem; line-height: 1.6; color: #f1f5f9; white-space: pre-wrap; word-break: break-word; }

        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem; }
        .metric-card { background: #0b111c; border: 1px solid var(--border); border-radius: 10px; padding: 1rem; text-align: center; }
        .metric-value { font-size: 1.5rem; font-weight: 700; color: #fdba74; margin-top: 0.25rem; }
        .metric-label { font-size: 0.85rem; color: var(--text-sub); }

        .voice-controls { display: flex; gap: 1rem; align-items: center; margin-top: 1rem; }
        .footer { margin-top: 2rem; color: var(--text-sub); font-size: 0.85rem; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>தமிழ் நுண்ணிய மாதிரி (Tamil Nano LLM)</h1>
        <p>Pretrained & Fine-Tuned Multi-Task AI Engine on Apple Silicon M4</p>
        <div class="badge-row">
            <div class="badge"><span class="status-dot"></span> M4 MPS Accelerated</div>
            <div class="badge accent">12.6M Transformer</div>
            <div class="badge">ChatML Dialogue</div>
            <div class="badge accent">8,192 BPE Tamil Vocab</div>
        </div>
    </div>

    <div class="container">
        <div class="tabs">
            <button class="tab-btn active" onclick="openTab(event, 'tab-chat')">💬 உரையாடல் (Chat)</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-translate')">🌐 மொழியாக்கம் (Translate)</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-paraphrase')">🔄 மாற்றுரை (Paraphrase)</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-mining')">🔍 உரைச் சுரங்கம் (NER)</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-voice')">🎙️ குரல் உதவியாளர் (Voice)</button>
            <button class="tab-btn" onclick="openTab(event, 'tab-metrics')">📊 விவரங்கள் (Metrics)</button>
        </div>

        <!-- Tab 1: Chatbot -->
        <div id="tab-chat" class="tab-content active">
            <div class="quick-samples">
                <span class="sample-pill" onclick="setSample('chat-input', 'வணக்கம்! நீங்கள் யார்?')">வணக்கம்! நீங்கள் யார்?</span>
                <span class="sample-pill" onclick="setSample('chat-input', 'தமிழ் மொழியின் சிறப்புகள் பற்றி சுருக்கமாக கூறுங்கள்.')">தமிழ் மொழியின் சிறப்புகள்</span>
                <span class="sample-pill" onclick="setSample('chat-input', 'செயற்கை நுண்ணறிவு (AI) என்றால் என்ன?')">செயற்கை நுண்ணறிவு என்றால் என்ன?</span>
            </div>
            <div class="form-group">
                <label for="chat-input">உங்கள் கேள்வி அல்லது செய்தியை உள்ளிடவும்:</label>
                <textarea id="chat-input" rows="3" placeholder="எ.கா: தமிழ் மொழியின் சிறப்புகள் பற்றி சுருக்கமாக கூறுங்கள்."></textarea>
            </div>
            <button class="btn" onclick="submitTask('chat', 'chat-input', 'chat-res')">விடை பெறுக (Generate)</button>
            <div class="result-box" id="chat-res-box" style="display:none;">
                <h4>மாதிரியின் பதில் (Model Response)</h4>
                <div class="result-text" id="chat-res"></div>
            </div>
        </div>

        <!-- Tab 2: Translation -->
        <div id="tab-translate" class="tab-content">
            <div class="form-group">
                <label>மொழியாக்க வகை (Direction):</label>
                <select id="trans-type">
                    <option value="translation_en_ta">English ➔ Tamil (ஆங்கிலம் -> தமிழ்)</option>
                    <option value="translation_ta_en">Tamil ➔ English (தமிழ் -> ஆங்கிலம்)</option>
                </select>
            </div>
            <div class="quick-samples">
                <span class="sample-pill" onclick="setSample('trans-input', 'Knowledge is power.')">Knowledge is power.</span>
                <span class="sample-pill" onclick="setSample('trans-input', 'Artificial intelligence helps solve complex problems.')">AI helps solve problems.</span>
                <span class="sample-pill" onclick="setSample('trans-input', 'வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?')">வணக்கம், எப்படி இருக்கிறீர்கள்?</span>
            </div>
            <div class="form-group">
                <label for="trans-input">மூல உரை (Source Text):</label>
                <textarea id="trans-input" rows="3" placeholder="Enter sentence to translate..."></textarea>
            </div>
            <button class="btn" onclick="submitTask(document.getElementById('trans-type').value, 'trans-input', 'trans-res')">மொழியாக்கம் செய் (Translate)</button>
            <div class="result-box" id="trans-res-box" style="display:none;">
                <h4>மொழிபெயர்ப்பு முடிவு (Translation Result)</h4>
                <div class="result-text" id="trans-res"></div>
            </div>
        </div>

        <!-- Tab 3: Paraphrasing -->
        <div id="tab-paraphrase" class="tab-content">
            <div class="quick-samples">
                <span class="sample-pill" onclick="setSample('para-input', 'மழை பெய்ததால் விளையாட்டு போட்டி ஒத்திவைக்கப்பட்டது.')">மழை பெய்ததால் போட்டி ஒத்திவைப்பு</span>
                <span class="sample-pill" onclick="setSample('para-input', 'தாங்கள் எப்போது அலுவலகத்திற்கு வருகை புரிவீர்கள்?')">அலுவலக வருகை (பேச்சு வழக்கு)</span>
            </div>
            <div class="form-group">
                <label for="para-input">மறுவடிவம் செய்ய வேண்டிய வாக்கியம்:</label>
                <textarea id="para-input" rows="3" placeholder="எ.கா: மழை பெய்ததால் விளையாட்டு போட்டி ஒத்திவைக்கப்பட்டது."></textarea>
            </div>
            <button class="btn" onclick="submitTask('paraphrase', 'para-input', 'para-res')">மாற்றுரை அமை (Paraphrase)</button>
            <div class="result-box" id="para-res-box" style="display:none;">
                <h4>மாற்று வாக்கியம் (Paraphrased Output)</h4>
                <div class="result-text" id="para-res"></div>
            </div>
        </div>

        <!-- Tab 4: Text Mining & NER -->
        <div id="tab-mining" class="tab-content">
            <div class="quick-samples">
                <span class="sample-pill" onclick="setSample('ner-input', 'அப்துல் கலாம் ராமேஸ்வரத்தில் பிறந்தார்.')">அப்துல் கலாம் ராமேஸ்வரம்</span>
                <span class="sample-pill" onclick="setSample('ner-input', 'சுந்தர் பிச்சை சென்னையில் பிறந்து தற்போது கூகுள் தலைமை நிர்வாகியாக உள்ளார்.')">சுந்தர் பிச்சை கூகுள்</span>
                <span class="sample-pill" onclick="setSample('ner-input', 'இஸ்ரோ நிறுவனம் ஸ்ரீஹரிகோட்டாவிலிருந்து சந்திரயான் விண்கலத்தை செலுத்தியது.')">இஸ்ரோ ஸ்ரீஹரிகோட்டா</span>
            </div>
            <div class="form-group">
                <label for="ner-input">பகுப்பாய்வு செய்ய வேண்டிய உரை:</label>
                <textarea id="ner-input" rows="3" placeholder="எ.கா: சுந்தர் பிச்சை சென்னையில் பிறந்து தற்போது கூகுள் தலைமை நிர்வாகியாக உள்ளார்."></textarea>
            </div>
            <button class="btn" onclick="submitTask('ner', 'ner-input', 'ner-res')">பிரித்தெடு (Extract Entities)</button>
            <div class="result-box" id="ner-res-box" style="display:none;">
                <h4>பிரித்தெடுக்கப்பட்ட விவரங்கள் (Extracted Entities)</h4>
                <div class="result-text" id="ner-res"></div>
            </div>
        </div>

        <!-- Tab 5: Voice Assistant -->
        <div id="tab-voice" class="tab-content">
            <p style="color: var(--text-sub); margin-bottom: 1rem;">மைக்ரோஃபோன் மூலம் தமிழில் பேசி உடனடி குரல் பதிலை பெறுங்கள் (Speech-to-Text & Text-to-Speech).</p>
            <div class="voice-controls">
                <button class="btn" id="mic-btn" onclick="toggleVoiceInput()">🎙️ பேச்சைத் தொடங்கு (Start Listening)</button>
                <button class="btn secondary" onclick="speakText(document.getElementById('voice-res').innerText)">🔊 பதிலை ஒலிபரப்பு (Speak)</button>
            </div>
            <div class="result-box" id="voice-res-box" style="margin-top: 1.5rem;">
                <h4>கேட்ட உரை (Recognized Speech)</h4>
                <div class="result-text" id="voice-user-text" style="color: #cbd5e1; margin-bottom: 1rem;">[பேசுவதற்கு மேலே உள்ள பொத்தானை அழுத்தவும்]</div>
                <h4>மாதிரியின் குரல் பதில் (Voice Assistant Response)</h4>
                <div class="result-text" id="voice-res">---</div>
            </div>
        </div>

        <!-- Tab 6: Metrics & Architecture -->
        <div id="tab-metrics" class="tab-content">
            <h3 style="color: #fdba74; margin-bottom: 1rem;">மாதிரி வடிவமைப்பு & செயல்திறன் குறியீடுகள்</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Parameters (அளவு)</div>
                    <div class="metric-value">12.6M</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Layers / Dim</div>
                    <div class="metric-value">6 L / 384 D</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Attention Mechanism</div>
                    <div class="metric-value">GQA + RoPE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Activation Function</div>
                    <div class="metric-value">SwiGLU</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Tamil Vocab Size</div>
                    <div class="metric-value">8,192 BPE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Hardware Device</div>
                    <div class="metric-value">Apple M4 MPS</div>
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        Tamil Nano LLM Engine &bull; Developed natively on Apple Silicon Mac Mini M4
    </div>

    <script>
        function openTab(evt, tabName) {
            var contents = document.getElementsByClassName("tab-content");
            for (var i = 0; i < contents.length; i++) contents[i].classList.remove("active");
            var buttons = document.getElementsByClassName("tab-btn");
            for (var i = 0; i < buttons.length; i++) buttons[i].classList.remove("active");
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }

        function setSample(elemId, text) {
            document.getElementById(elemId).value = text;
        }

        async function submitTask(task, inputId, resultId) {
            const text = document.getElementById(inputId).value.trim();
            if (!text) return;
            
            const resBox = document.getElementById(resultId + "-box");
            const resElem = document.getElementById(resultId);
            resBox.style.display = "block";
            resElem.innerText = "உருவாக்குகிறது... (Generating...)";

            try {
                const response = await fetch("/api/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text, task: task, temperature: 0.2 })
                });
                const data = await response.json();
                resElem.innerText = data.response || "No response generated.";
            } catch (err) {
                resElem.innerText = "பிழை ஏற்பட்டது: " + err.message;
            }
        }

        // Web Speech Recognition & Synthesis for Voice AI Tab
        let recognition = null;
        let isListening = false;

        function toggleVoiceInput() {
            const btn = document.getElementById("mic-btn");
            const userTextElem = document.getElementById("voice-user-text");
            const resElem = document.getElementById("voice-res");

            if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
                alert("Speech recognition is not supported in this browser. Please use Google Chrome or Safari.");
                return;
            }

            if (isListening) {
                if (recognition) recognition.stop();
                isListening = false;
                btn.innerText = "🎙️ பேச்சைத் தொடங்கு (Start Listening)";
                return;
            }

            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRec();
            recognition.lang = 'ta-IN';
            recognition.interimResults = false;

            recognition.onstart = () => {
                isListening = true;
                btn.innerText = "🛑 கேட்பதை நிறுத்து (Stop)";
                userTextElem.innerText = "கேட்கிறது... (Listening in Tamil...)";
            };

            recognition.onresult = async (event) => {
                const spokenText = event.results[0][0].transcript;
                userTextElem.innerText = spokenText;
                resElem.innerText = "பதிலை உருவாக்குகிறது... (Generating answer...)";

                const response = await fetch("/api/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: spokenText, task: "chat", temperature: 0.2 })
                });
                const data = await response.json();
                resElem.innerText = data.response || "";
                speakText(data.response);
            };

            recognition.onerror = (e) => {
                userTextElem.innerText = "குரல் கண்டறிவதில் பிழை: " + e.error;
                isListening = false;
                btn.innerText = "🎙️ பேச்சைத் தொடங்கு (Start Listening)";
            };

            recognition.onend = () => {
                isListening = false;
                btn.innerText = "🎙️ பேச்சைத் தொடங்கு (Start Listening)";
            };

            recognition.start();
        }

        function speakText(text) {
            if (!text || text === "---") return;
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'ta-IN';
                window.speechSynthesis.speak(utterance);
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
