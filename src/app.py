import os
import sys
import streamlit as st
from datetime import datetime
import random
import string

# Direct path routing from src folder to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.rag import query_rag_engine
import ollama
import chromadb
from chromadb.utils import embedding_functions

# ─── Voice helpers (graceful fallback if libs missing) ────────────────────────
try:
    import speech_recognition as sr
    import pyttsx3

    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False


def speak_text(text: str):
    if not VOICE_AVAILABLE:
        st.warning("pyttsx3 / speech_recognition not installed.")
        return
    try:
        clean = text.replace("*", "").replace("#", "").replace("-", " ")
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(clean)
        engine.runAndWait()
    except Exception as e:
        st.error(f"TTS Error: {e}")


def listen_to_microphone() -> str:
    if not VOICE_AVAILABLE:
        st.warning("pyttsx3 / speech_recognition not installed.")
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.toast("🎙️ Listening… Speak your medical question now.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            st.toast("✨ Processing audio…")
            return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("⏱️ No audio detected.")
    except sr.UnknownValueError:
        st.error("❌ Audio unclear. Please try again.")
    except Exception as e:
        st.error(f"Audio error: {e}")
    return ""


# ─── Medicine Search ───────────────────────────────────────────────────────────
def search_medicine_details(med_name: str) -> str:
    """Searches ChromaDB vector store for medicine and returns structured clinical profile."""
    db_storage_path = os.path.join(project_root, "chroma_db")
    client = chromadb.PersistentClient(path=db_storage_path)
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    try:
        collection = client.get_collection(
            name="medical_guide_collection", embedding_function=embedding_model
        )
        results = collection.query(query_texts=[med_name], n_results=2)

        # FIX: Flatten nested list returned by ChromaDB
        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            if isinstance(results["documents"][0], list):
                retrieved_chunks = results["documents"][0]
            else:
                retrieved_chunks = results["documents"]

        context_str = "\n\n".join(retrieved_chunks)

        system_prompt = (
            f"You are a clinical pharmacology assistant. Analyze the reference material regarding '{med_name}'. "
            "Provide a clear, clinical summary strictly structured into three exact headings:\n"
            "### 📋 Uses\n"
            "### ⚠️ Side Effects & Warnings\n"
            "### 💊 Dosage Information\n\n"
            "If the reference material does not contain specific information for a section, use your "
            "comprehensive internal medical knowledge to fill in standard clinical guidelines for that medicine. "
            "Never use phrases like 'based on the text' or 'the chunks state'."
        )
        user_prompt = (
            f"Reference Material:\n{context_str}\n\nMedicine to Look Up:\n{med_name}"
        )

        response = ollama.generate(
            model="llama3",
            system=system_prompt,
            prompt=user_prompt,
            options={"temperature": 0.1},
        )
        return response["response"]
    except Exception as e:
        return f"[!] Error searching knowledge base: {str(e)}"


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedCore AI — Diagnostic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "session_id" not in st.session_state:
    st.session_state.session_id = "MC-" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=8)
    )
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "chat"
if "med_result" not in st.session_state:
    st.session_state.med_result = None
if "med_name_searched" not in st.session_state:
    st.session_state.med_name_searched = ""
if "voice_buffer" not in st.session_state:
    st.session_state.voice_buffer = ""
# Key counters — incrementing forces Streamlit to re-render the text_input fresh (clears it)
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0
if "med_input_key" not in st.session_state:
    st.session_state.med_input_key = 0


# ─── Loading Overlay ──────────────────────────────────────────────────────────
def show_loader(label: str = "Processing", sublabel: str = "Accessing knowledge base…"):
    st.markdown(
        f"""
    <style>
    #medcore-loader {{
        position:fixed; inset:0; z-index:99999;
        background:rgba(2,13,26,0.93);
        backdrop-filter:blur(6px);
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        gap:24px; animation:ldr-in 0.2s ease-out;
    }}
    @keyframes ldr-in {{ from{{opacity:0}} to{{opacity:1}} }}
    .ldr-ring-wrap {{ position:relative; width:90px; height:90px; }}
    .ldr-ring {{ position:absolute; inset:0; border-radius:50%; border:2px solid transparent; }}
    .ldr-ring.r1 {{ border-top-color:#00d2c8; animation:spin 1.1s linear infinite; }}
    .ldr-ring.r2 {{ inset:10px; border-right-color:rgba(0,210,200,0.4); animation:spin 0.75s linear infinite reverse; }}
    .ldr-ring.r3 {{ inset:20px; border-bottom-color:rgba(0,210,200,0.2); animation:spin 1.6s linear infinite; }}
    .ldr-core {{ position:absolute; inset:32px; background:radial-gradient(circle,rgba(0,210,200,0.25),transparent); border-radius:50%; animation:core-pulse 1.2s ease-in-out infinite; }}
    @keyframes spin {{ to{{ transform:rotate(360deg); }} }}
    @keyframes core-pulse {{ 0%,100%{{opacity:0.4;transform:scale(0.9)}} 50%{{opacity:1;transform:scale(1.1)}} }}
    .ldr-label {{ font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:500; letter-spacing:3px; text-transform:uppercase; color:#00d2c8; text-align:center; }}
    .ldr-sublabel {{ font-size:13px; color:#3a7a9a; text-align:center; }}
    .ldr-bar-track {{ width:260px; height:2px; background:rgba(0,210,200,0.1); border-radius:2px; overflow:hidden; }}
    .ldr-bar-fill {{ height:100%; width:40%; background:linear-gradient(90deg,transparent,#00d2c8,transparent); animation:bar-sweep 1.4s ease-in-out infinite; }}
    @keyframes bar-sweep {{ 0%{{transform:translateX(-200%)}} 100%{{transform:translateX(500%)}} }}
    .ldr-dots {{ display:flex; gap:8px; }}
    .ldr-dot {{ width:5px; height:5px; border-radius:50%; background:rgba(0,210,200,0.3); animation:dot-pop 1.2s ease-in-out infinite; }}
    .ldr-dot:nth-child(2){{ animation-delay:0.2s; }}
    .ldr-dot:nth-child(3){{ animation-delay:0.4s; }}
    @keyframes dot-pop {{ 0%,80%,100%{{transform:scale(1);opacity:0.3}} 40%{{transform:scale(1.6);opacity:1;background:#00d2c8}} }}
    </style>
    <div id="medcore-loader">
      <div class="ldr-ring-wrap">
        <div class="ldr-ring r1"></div><div class="ldr-ring r2"></div>
        <div class="ldr-ring r3"></div><div class="ldr-core"></div>
      </div>
      <div class="ldr-label">{label}</div>
      <div class="ldr-sublabel">{sublabel}</div>
      <div class="ldr-bar-track"><div class="ldr-bar-fill"></div></div>
      <div class="ldr-dots"><div class="ldr-dot"></div><div class="ldr-dot"></div><div class="ldr-dot"></div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;}
html,body,[data-testid="stAppViewContainer"]{background:#020d1a!important;color:#c8dff0!important;font-family:'Space Grotesk',sans-serif!important;}
[data-testid="stAppViewContainer"]{padding:0!important;}
[data-testid="stHeader"]{display:none!important;}
[data-testid="stToolbar"]{display:none!important;}
section[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
footer{display:none!important;}
[data-testid="stDecoration"]{display:none!important;}
[data-testid="stTabs"]{display:none!important;}

/* ── Header ── */
.medcore-header{background:linear-gradient(135deg,#020d1a 0%,#031a2e 40%,#042338 100%);border-bottom:1px solid rgba(0,210,200,0.2);padding:24px 48px 18px;position:relative;overflow:hidden;}
.medcore-header::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:radial-gradient(ellipse 60% 80% at 85% 50%,rgba(0,180,200,0.06) 0%,transparent 70%),radial-gradient(ellipse 30% 60% at 10% 50%,rgba(0,120,255,0.05) 0%,transparent 70%);pointer-events:none;}
.header-grid{display:grid;grid-template-columns:1fr auto;align-items:center;gap:24px;position:relative;z-index:1;}
.brand-eyebrow{font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:3px;color:#00c8be;text-transform:uppercase;margin-bottom:6px;opacity:0.8;}
.brand-name{font-size:30px;font-weight:700;letter-spacing:-0.5px;color:#e8f4ff;line-height:1;}
.brand-name span{color:#00d2c8;}
.brand-tagline{font-size:12px;font-weight:300;color:#6a96b8;margin-top:5px;}
.vitals-strip{display:flex;gap:16px;align-items:center;}
.vital-card{background:rgba(0,210,200,0.05);border:1px solid rgba(0,210,200,0.15);border-radius:8px;padding:9px 14px;text-align:center;min-width:84px;}
.vital-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2px;color:#4a8aaa;text-transform:uppercase;margin-bottom:4px;}
.vital-value{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:500;color:#00d2c8;}
.vital-value.green{color:#00e676;}
.vital-value.amber{color:#ffc107;font-size:12px;}

/* ── Mode switcher tabs ── */
.mode-btn-wrap .stButton>button{background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;border-radius:0!important;color:#2a5a7a!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important;letter-spacing:2px!important;text-transform:uppercase!important;padding:13px 28px!important;width:auto!important;box-shadow:none!important;transition:color 0.2s!important;}
.mode-btn-wrap .stButton>button:hover{color:#6aaac8!important;background:transparent!important;box-shadow:none!important;transform:none!important;}
.mode-btn-wrap.active .stButton>button{color:#00d2c8!important;border-bottom:2px solid #00d2c8!important;}

/* ── Live biomonitor bar ── */
.vitals-graph-bar{background:linear-gradient(90deg,#020d1a,#031625,#020d1a);border-bottom:1px solid rgba(0,210,200,0.12);padding:10px 48px;display:flex;align-items:center;gap:16px;overflow:hidden;}
.vgb-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2.5px;color:#1a5a7a;text-transform:uppercase;white-space:nowrap;flex-shrink:0;}
.vgb-canvas-wrap{flex:1;height:48px;}
canvas.vitals-canvas{width:100%;height:48px;display:block;}
.vgb-legend{display:flex;gap:14px;flex-shrink:0;}
.vgb-leg-item{display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#3a7a9a;}
.vgb-leg-dot{width:8px;height:2px;border-radius:1px;}

/* ── Left panel ── */
.left-panel{background:#020d1a;border-right:1px solid rgba(0,210,200,0.1);padding:22px 18px;overflow-y:auto;display:flex;flex-direction:column;gap:20px;}
.panel-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:3px;color:#2a5a7a;text-transform:uppercase;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid rgba(0,210,200,0.08);}
.status-row{display:flex;align-items:center;gap:10px;padding:9px 11px;border-radius:8px;background:rgba(0,230,118,0.05);border:1px solid rgba(0,230,118,0.1);margin-bottom:7px;}
.dot{width:7px;height:7px;border-radius:50%;background:#00e676;box-shadow:0 0 6px #00e676;flex-shrink:0;animation:pulse-dot 2s ease-in-out infinite;}
.dot.amber{background:#ffc107;box-shadow:0 0 6px #ffc107;}
@keyframes pulse-dot{0%,100%{opacity:1;}50%{opacity:0.5;}}
.status-text{font-size:12px;color:#a0c8e0;font-weight:500;}
.cap-item{display:flex;align-items:flex-start;gap:10px;padding:10px 11px;border-radius:8px;border:1px solid rgba(0,210,200,0.08);background:rgba(255,255,255,0.02);margin-bottom:6px;transition:border-color 0.2s,background 0.2s;}
.cap-item:hover{border-color:rgba(0,210,200,0.2);background:rgba(0,210,200,0.04);}
.cap-icon{font-size:14px;flex-shrink:0;margin-top:1px;}
.cap-title{font-size:12px;font-weight:600;color:#c0dff0;margin-bottom:2px;}
.cap-desc{font-size:11px;color:#4a7a9a;line-height:1.4;}
.quick-btn{display:block;width:100%;background:rgba(0,210,200,0.04);border:1px solid rgba(0,210,200,0.12);border-radius:7px;padding:9px 11px;font-size:12px;color:#7ab4cc;cursor:pointer;text-align:left;font-family:'Space Grotesk',sans-serif;transition:all 0.2s;line-height:1.4;margin-bottom:5px;}
.quick-btn:hover{background:rgba(0,210,200,0.09);border-color:rgba(0,210,200,0.3);color:#b0d8f0;}

/* ── Center panel ── */
.console-header{padding:14px 26px;border-bottom:1px solid rgba(0,210,200,0.08);display:flex;align-items:center;justify-content:space-between;flex-shrink:0;}
.console-title{font-family:'JetBrains Mono',monospace;font-size:11px;color:#2a6a8a;letter-spacing:2px;text-transform:uppercase;}
.console-meta{font-family:'JetBrains Mono',monospace;font-size:11px;color:#1a4a6a;}

/* Messages */
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:60px 40px;text-align:center;min-height:300px;position:relative;}
.grid-bg{position:absolute;inset:0;background-image:linear-gradient(rgba(0,210,200,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,210,200,0.03) 1px,transparent 1px);background-size:32px 32px;pointer-events:none;}
.empty-icon{font-size:44px;opacity:0.25;}
.empty-title{font-size:18px;font-weight:600;color:#3a6a8a;}
.empty-sub{font-size:13px;color:#1a4a6a;max-width:300px;line-height:1.6;}
.msg-row{display:flex;gap:12px;align-items:flex-start;margin-bottom:18px;animation:msg-in 0.3s ease-out;}
@keyframes msg-in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg-row.user-row{flex-direction:row-reverse;}
.msg-avatar{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;}
.msg-avatar.ai-av{background:linear-gradient(135deg,rgba(0,210,200,0.15),rgba(0,120,255,0.1));border:1px solid rgba(0,210,200,0.2);}
.msg-avatar.user-av{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);}
.msg-meta{font-family:'JetBrains Mono',monospace;font-size:9px;color:#2a5a7a;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:5px;}
.msg-bubble{max-width:76%;padding:13px 16px;border-radius:12px;font-size:14px;line-height:1.7;}
.msg-bubble.ai{background:rgba(0,30,50,0.8);border:1px solid rgba(0,210,200,0.15);color:#c0dff0;border-top-left-radius:3px;}
.msg-bubble.user{background:rgba(0,80,150,0.15);border:1px solid rgba(0,120,255,0.2);color:#d0e8ff;border-top-right-radius:3px;}
.src-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(0,210,200,0.07);border:1px solid rgba(0,210,200,0.15);border-radius:5px;padding:4px 10px;font-family:'JetBrains Mono',monospace;font-size:10px;color:#00d2c8;margin-top:10px;}

/* Medicine result */
.med-intro{margin-bottom:16px;padding:16px 20px;background:rgba(0,210,200,0.04);border:1px solid rgba(0,210,200,0.1);border-radius:10px;font-size:13px;color:#5a9ab8;line-height:1.6;}
.med-intro strong{color:#00d2c8;display:block;margin-bottom:4px;font-size:11px;letter-spacing:1.5px;text-transform:uppercase;}
.med-result-card{background:rgba(0,20,40,0.6);border:1px solid rgba(0,210,200,0.18);border-radius:12px;padding:24px 26px;margin-top:16px;}
.med-result-title{font-family:'JetBrains Mono',monospace;font-size:13px;letter-spacing:2px;text-transform:uppercase;color:#00d2c8;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid rgba(0,210,200,0.12);}
.med-result-body{font-size:14px;color:#b0d0e8;line-height:1.8;}

/* ── Input zone ── */
.input-zone-label{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:2.5px;color:#2a5a7a;text-transform:uppercase;margin-bottom:6px;}

/* Streamlit input styling */
.stTextInput>div>div>input{background:rgba(0,20,40,0.6)!important;border:1px solid rgba(0,210,200,0.2)!important;border-radius:10px!important;color:#c8e8ff!important;font-family:'Space Grotesk',sans-serif!important;font-size:14px!important;padding:13px 16px!important;caret-color:#00d2c8!important;}
.stTextInput>div>div>input:focus{border-color:rgba(0,210,200,0.5)!important;box-shadow:0 0 0 2px rgba(0,210,200,0.08)!important;outline:none!important;}
.stTextInput>div>div>input::placeholder{color:#1a4a6a!important;}
.stTextInput>label{display:none!important;}

/* Primary action buttons */
.stButton>button{background:linear-gradient(135deg,#006060,#004a80)!important;border:1px solid rgba(0,210,200,0.35)!important;border-radius:10px!important;color:#c8f8f8!important;font-family:'Space Grotesk',sans-serif!important;font-size:13px!important;font-weight:600!important;letter-spacing:0.5px!important;padding:13px 22px!important;width:100%!important;transition:all 0.2s!important;}
.stButton>button:hover{background:linear-gradient(135deg,#007070,#005a90)!important;border-color:rgba(0,210,200,0.55)!important;box-shadow:0 0 16px rgba(0,210,200,0.15)!important;transform:translateY(-1px)!important;}
.stButton>button:active{transform:translateY(0)!important;}

/* Clear / secondary buttons */
.clear-wrap .stButton>button{background:rgba(255,50,80,0.06)!important;border-color:rgba(255,50,80,0.2)!important;color:#aa4050!important;font-size:12px!important;padding:11px 14px!important;box-shadow:none!important;}
.clear-wrap .stButton>button:hover{background:rgba(255,50,80,0.12)!important;box-shadow:none!important;transform:none!important;}

/* Voice button */
.voice-wrap .stButton>button{background:rgba(0,150,255,0.07)!important;border-color:rgba(0,150,255,0.25)!important;color:#60aaff!important;font-size:12px!important;padding:11px 14px!important;box-shadow:none!important;}
.voice-wrap .stButton>button:hover{background:rgba(0,150,255,0.14)!important;box-shadow:0 0 10px rgba(0,150,255,0.1)!important;transform:none!important;}

/* Read aloud button */
.aloud-wrap .stButton>button{background:rgba(0,230,118,0.05)!important;border-color:rgba(0,230,118,0.2)!important;color:#40cc80!important;font-size:12px!important;padding:11px 14px!important;box-shadow:none!important;}
.aloud-wrap .stButton>button:hover{background:rgba(0,230,118,0.1)!important;box-shadow:none!important;transform:none!important;}

/* ── Right panel ── */
.right-panel{background:#020d1a;border-left:1px solid rgba(0,210,200,0.1);padding:22px 18px;overflow-y:auto;display:flex;flex-direction:column;gap:18px;}
.stat-block{padding:14px 16px;background:rgba(255,255,255,0.02);border:1px solid rgba(0,210,200,0.08);border-radius:10px;}
.stat-number{font-family:'JetBrains Mono',monospace;font-size:26px;font-weight:500;color:#00d2c8;line-height:1;margin:5px 0 3px;}
.stat-label{font-size:11px;color:#4a7a9a;}
.disclaimer-box{padding:13px 14px;background:rgba(255,150,0,0.04);border:1px solid rgba(255,150,0,0.12);border-radius:10px;font-size:11px;color:#7a6040;line-height:1.6;}
.disclaimer-box strong{color:#aa8050;display:block;margin-bottom:4px;font-size:10px;letter-spacing:1px;text-transform:uppercase;}
.history-item{padding:8px 11px;background:rgba(0,210,200,0.03);border:1px solid rgba(0,210,200,0.07);border-radius:8px;font-size:11px;color:#3a6a8a;font-family:'JetBrains Mono',monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:5px;}

@media(max-width:1100px){.right-panel{display:none!important;}}
@media(max-width:800px){.left-panel{display:none!important;}.vitals-strip{display:none!important;}.medcore-header{padding:18px 20px 14px;}}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="medcore-header">
  <div class="header-grid">
    <div>
      <div class="brand-eyebrow">DIAGNOSTIC INTELLIGENCE SYSTEM · v2.1</div>
      <div class="brand-name">Med<span>Core</span> AI</div>
      <div class="brand-tagline">Evidence-based clinical reference · Local RAG engine · Private by design</div>
    </div>
    <div class="vitals-strip">
      <div class="vital-card"><div class="vital-label">RAG Engine</div><div class="vital-value green">LIVE</div></div>
      <div class="vital-card"><div class="vital-label">Queries</div><div class="vital-value">{st.session_state.query_count:03d}</div></div>
      <div class="vital-card"><div class="vital-label">Session</div><div class="vital-value amber">{st.session_state.session_id}</div></div>
      <div class="vital-card" style="padding:8px 12px;">
        <div class="vital-label">Signal</div>
        <svg viewBox="0 0 120 38" width="100" height="32" xmlns="http://www.w3.org/2000/svg">
          <polyline points="0,19 18,19 26,7 32,31 38,19 58,19 66,3 72,35 78,19 120,19"
            fill="none" stroke="#00d2c8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>
        </svg>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── Mode Switcher ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"]:has(.mode-btn-wrap){
    background:#020d1a;border-bottom:1px solid rgba(0,210,200,0.1);padding:0 48px;gap:0!important;
}
div[data-testid="column"]:has(.mode-btn-wrap){flex:0 0 auto!important;width:auto!important;padding:0!important;}
</style>
""",
    unsafe_allow_html=True,
)

mode_col1, mode_col2, mode_spacer = st.columns([1, 1, 8], gap="small")
with mode_col1:
    cls = "active" if st.session_state.active_mode == "chat" else ""
    st.markdown(f'<div class="mode-btn-wrap {cls}">', unsafe_allow_html=True)
    if st.button("◈  Clinical Chat AI", key="mode_chat"):
        st.session_state.active_mode = "chat"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
with mode_col2:
    cls = "active" if st.session_state.active_mode == "med" else ""
    st.markdown(f'<div class="mode-btn-wrap {cls}">', unsafe_allow_html=True)
    if st.button("⬡  Medicine Search", key="mode_med"):
        st.session_state.active_mode = "med"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Live Biomonitor Graph ─────────────────────────────────────────────────────
st.markdown(
    """
<div class="vitals-graph-bar">
  <div class="vgb-label">◈ Live Biomonitor</div>
  <div class="vgb-canvas-wrap">
    <canvas class="vitals-canvas" id="vitalsChart"></canvas>
  </div>
  <div class="vgb-legend">
    <div class="vgb-leg-item"><div class="vgb-leg-dot" style="background:#00d2c8;"></div>HR</div>
    <div class="vgb-leg-item"><div class="vgb-leg-dot" style="background:#00e676;"></div>SpO₂</div>
    <div class="vgb-leg-item"><div class="vgb-leg-dot" style="background:#ffc107;"></div>BP</div>
  </div>
</div>
<script>
(function(){
  const canvas=document.getElementById('vitalsChart');
  if(!canvas)return;
  const ctx=canvas.getContext('2d');
  function resize(){canvas.width=canvas.offsetWidth*devicePixelRatio;canvas.height=canvas.offsetHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);}
  resize();
  const W=()=>canvas.offsetWidth, H=()=>canvas.offsetHeight, N=120;
  const wave=(b,a,f,n)=>Array.from({length:N},(_,i)=>b+a*Math.sin(i*f)+(Math.random()-0.5)*n);
  let hr=wave(0.5,0.28,0.18,0.06),spo2=wave(0.75,0.12,0.09,0.03),bp=wave(0.35,0.20,0.13,0.05),t=N;
  const push=(arr,b,a,f,n)=>{arr.push(b+a*Math.sin(t*f)+(Math.random()-0.5)*n);if(arr.length>N)arr.shift();};
  function line(data,color,lw){
    const w=W(),h=H(),step=w/(N-1);
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.lineJoin='round';ctx.lineCap='round';
    data.forEach((v,i)=>{const x=i*step,y=h-v*h;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.stroke();
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=lw+3;ctx.globalAlpha=0.12;
    data.forEach((v,i)=>{const x=i*step,y=h-v*h;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});ctx.stroke();ctx.globalAlpha=1;
  }
  function frame(){
    const w=W(),h=H();ctx.clearRect(0,0,w,h);
    ctx.strokeStyle='rgba(0,210,200,0.06)';ctx.lineWidth=1;
    for(let i=0;i<=4;i++){const y=(i/4)*h;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
    for(let i=0;i<=8;i++){const x=(i/8)*w;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
    const g=ctx.createLinearGradient(w-24,0,w,0);g.addColorStop(0,'rgba(0,210,200,0)');g.addColorStop(1,'rgba(0,210,200,0.22)');
    ctx.fillStyle=g;ctx.fillRect(w-24,0,24,h);
    line(spo2,'#00e676',1.2);line(bp,'#ffc107',1.2);line(hr,'#00d2c8',1.8);
    push(hr,0.5,0.28,0.18,0.06);push(spo2,0.75,0.12,0.09,0.03);push(bp,0.35,0.20,0.13,0.05);t++;
    requestAnimationFrame(frame);
  }
  window.addEventListener('resize',resize);frame();
})();
</script>
""",
    unsafe_allow_html=True,
)

# ─── 3-Column Layout ──────────────────────────────────────────────────────────
left_col, center_col, right_col = st.columns([1.0, 3.4, 1.0], gap="small")

# ══════════════ LEFT PANEL ══════════════
with left_col:
    st.markdown(
        f"""
    <div class="left-panel">
      <div>
        <div class="panel-label">System Status</div>
        <div class="status-row"><div class="dot"></div><span class="status-text">RAG Engine Online</span></div>
        <div class="status-row" style="background:rgba(0,120,255,0.05);border-color:rgba(0,120,255,0.15);">
          <div class="dot amber"></div><span class="status-text">Knowledge Base Active</span>
        </div>
        <div class="status-row" style="background:rgba(255,193,7,0.05);border-color:rgba(255,193,7,0.12);">
          <div class="dot" style="background:#ffc107;box-shadow:0 0 6px #ffc107;animation:none;"></div>
          <span class="status-text">Llama3 Model Ready</span>
        </div>
        <div class="status-row" style="background:rgba(0,150,255,0.05);border-color:rgba(0,150,255,0.15);">
          <div class="dot" style="background:#60aaff;box-shadow:0 0 6px #60aaff;{'animation:pulse-dot 2s ease-in-out infinite;' if VOICE_AVAILABLE else 'animation:none;opacity:0.4;'}"></div>
          <span class="status-text">{'Voice Engine Ready' if VOICE_AVAILABLE else 'Voice Unavailable'}</span>
        </div>
      </div>
      <div>
        <div class="panel-label">Capabilities</div>
        <div class="cap-item"><div class="cap-icon">🔬</div><div><div class="cap-title">Symptom Analysis</div><div class="cap-desc">Differential assessment from clinical literature</div></div></div>
        <div class="cap-item"><div class="cap-icon">💊</div><div><div class="cap-title">Drug Reference</div><div class="cap-desc">Dosage, interactions & contraindications</div></div></div>
        <div class="cap-item"><div class="cap-icon">📋</div><div><div class="cap-title">Clinical Guidelines</div><div class="cap-desc">Evidence-based protocol lookup</div></div></div>
        <div class="cap-item"><div class="cap-icon">🎙️</div><div><div class="cap-title">Voice Input</div><div class="cap-desc">Speak questions, hear answers aloud</div></div></div>
      </div>
      <div>
        <div class="panel-label">Quick Queries</div>
        <div class="quick-btn">Signs of hypertensive emergency?</div>
        <div class="quick-btn">Normal HbA1c range for diabetics?</div>
        <div class="quick-btn">First-line treatment for UTI?</div>
        <div class="quick-btn">Acute appendicitis symptoms?</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════ CENTER PANEL ══════════════
with center_col:

    # ── CLINICAL CHAT MODE ────────────────────────────────────────────────────
    if st.session_state.active_mode == "chat":

        st.markdown(
            f"""
        <div class="console-header">
          <span class="console-title">◈ Diagnostic Console</span>
          <span class="console-meta">{st.session_state.session_id} · {len(st.session_state.messages)//2} exchanges</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Display messages or empty state
        if not st.session_state.messages:
            st.markdown(
                """
            <div class="empty-state">
              <div class="grid-bg"></div>
              <div class="empty-icon">🧬</div>
              <div class="empty-title">Ready for Consultation</div>
              <div class="empty-sub">Describe symptoms, ask about a condition, or use the 🎙️ voice button to speak your question.</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                ts = msg.get("time", "--:--")
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                    <div class="msg-row user-row">
                      <div class="msg-avatar user-av">👤</div>
                      <div>
                        <div class="msg-meta" style="text-align:right;">Patient Query · {ts}</div>
                        <div class="msg-bubble user">{msg["content"]}</div>
                      </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="msg-row">
                      <div class="msg-avatar ai-av">🧬</div>
                      <div>
                        <div class="msg-meta">MedCore AI · {ts}</div>
                        <div class="msg-bubble ai">{msg["content"]}<div class="src-badge">⬡ Retrieved from Local Knowledge Base</div></div>
                      </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

        # Input zone label
        st.markdown(
            '<div class="input-zone-label" style="padding:12px 0 6px;">◈ ENTER DIAGNOSTIC QUERY</div>',
            unsafe_allow_html=True,
        )

        # Text input — key is incremented to force clear after submit
        user_question = st.text_input(
            "query",
            label_visibility="collapsed",
            placeholder="Describe symptoms, ask about a drug, or reference a clinical condition…",
            key=f"chat_input_{st.session_state.chat_input_key}",
            value=st.session_state.voice_buffer,  # pre-fill from voice if available
        )

        # Buttons: Submit | Clear (stacked to avoid collision)
        submit = st.button(
            "⬡  Run Diagnostic Query", use_container_width=True, key="chat_submit"
        )

        btn_c1, btn_c2 = st.columns(2, gap="small")
        with btn_c1:
            st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
            clear = st.button(
                "✕ Clear History", use_container_width=True, key="chat_clear"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with btn_c2:
            st.markdown('<div class="voice-wrap">', unsafe_allow_html=True)
            voice_btn = st.button(
                "🎙️ Voice Input", use_container_width=True, key="chat_voice"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Handle voice input
        if voice_btn:
            spoken = listen_to_microphone()
            if spoken:
                st.session_state.voice_buffer = spoken
                st.rerun()

        # ── Handle clear
        if clear:
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.voice_buffer = ""
            st.session_state.chat_input_key += 1  # forces text input to re-render empty
            st.rerun()

        # ── Handle submit
        if submit and user_question.strip():
            ts = datetime.now().strftime("%H:%M")
            q = user_question.strip()

            st.session_state.messages.append({"role": "user", "content": q, "time": ts})
            st.session_state.voice_buffer = ""  # clear voice buffer
            st.session_state.chat_input_key += 1  # clears the text box on rerun

            show_loader("Running Diagnostic Query", "Scanning RAG knowledge base…")
            try:
                answer = query_rag_engine(q)
            except Exception as e:
                answer = f"⚠ Retrieval error: {str(e)}"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "time": datetime.now().strftime("%H:%M"),
                }
            )
            st.session_state.query_count += 1

            # Speak response asynchronously (non-blocking in background if possible)
            try:
                speak_text(answer)
            except Exception:
                pass

            st.rerun()

    # ── MEDICINE SEARCH MODE ──────────────────────────────────────────────────
    else:
        st.markdown(
            """
        <div class="console-header">
          <span class="console-title">⬡ Medicine Search — Clinical Profile Retrieval</span>
          <span class="console-meta">ChromaDB · Llama3</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="med-intro">
          <strong>◈ Knowledge Base Lookup</strong>
          Enter a generic or brand medicine name to retrieve its structured clinical profile — uses, side effects, warnings, and dosage — from your local vector database.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="input-zone-label" style="padding:8px 0 6px;">⬡ MEDICINE NAME</div>',
            unsafe_allow_html=True,
        )

        # Key increment forces input to clear when needed
        med_query = st.text_input(
            "med",
            label_visibility="collapsed",
            placeholder="e.g. Paracetamol, Metformin, Aconitum Napellus…",
            key=f"med_input_{st.session_state.med_input_key}",
        )

        # Primary action button
        med_submit = st.button(
            "⬡  Retrieve Clinical Profile", use_container_width=True, key="med_submit"
        )

        # Clear + Voice side by side (no collision — both same width)
        mc1, mc2 = st.columns(2, gap="small")
        with mc1:
            st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
            med_clear = st.button(
                "✕ Clear Results", use_container_width=True, key="med_clear"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with mc2:
            st.markdown('<div class="voice-wrap">', unsafe_allow_html=True)
            med_voice = st.button(
                "🎙️ Voice Input", use_container_width=True, key="med_voice"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Handle voice for medicine search
        if med_voice:
            spoken = listen_to_microphone()
            if spoken:
                st.session_state.med_name_searched = spoken
                st.session_state.med_input_key += 1
                st.rerun()

        # ── Handle clear — clears BOTH text box AND output
        if med_clear:
            st.session_state.med_result = None
            st.session_state.med_name_searched = ""
            st.session_state.med_input_key += 1  # forces input box to re-render blank
            st.rerun()

        # ── Handle submit
        if med_submit and med_query.strip():
            show_loader(
                "Retrieving Clinical Profile", f"Querying vector store for {med_query}…"
            )
            result = search_medicine_details(med_query.strip())
            st.session_state.med_result = result
            st.session_state.med_name_searched = med_query.strip()
            st.session_state.query_count += 1
            st.rerun()

        # ── Show result card
        if st.session_state.med_result:
            st.markdown(
                f"""
            <div class="med-result-card">
              <div class="med-result-title">💊 Clinical Profile — {st.session_state.med_name_searched.upper()}</div>
              <div class="med-result-body">{st.session_state.med_result.replace(chr(10), '<br>')}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Read aloud button below result
            st.markdown(
                '<div class="aloud-wrap" style="margin-top:10px;">',
                unsafe_allow_html=True,
            )
            if st.button(
                "🔊 Read Profile Aloud", use_container_width=True, key="med_aloud"
            ):
                speak_text(st.session_state.med_result)
            st.markdown("</div>", unsafe_allow_html=True)

# ══════════════ RIGHT PANEL ══════════════
with right_col:
    st.markdown(
        f"""
    <div class="right-panel">
      <div>
        <div class="panel-label">Session Metrics</div>
        <div class="stat-block">
          <div class="stat-label">Total Queries</div>
          <div class="stat-number">{st.session_state.query_count:03d}</div>
        </div>
        <div class="stat-block" style="margin-top:10px;">
          <div class="stat-label">Chat Exchanges</div>
          <div class="stat-number">{len(st.session_state.messages)//2:03d}</div>
        </div>
      </div>
      <div>
        <div class="panel-label">Active Mode</div>
        <div class="stat-block" style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#00d2c8;">
          {'◈ Clinical Chat' if st.session_state.active_mode == 'chat' else '⬡ Medicine Search'}
        </div>
      </div>
      <div>
        <div class="panel-label">Recent Queries</div>
    """,
        unsafe_allow_html=True,
    )

    all_queries = [
        m["content"] for m in st.session_state.messages if m["role"] == "user"
    ]
    if st.session_state.med_name_searched:
        all_queries.append(f"[MED] {st.session_state.med_name_searched}")
    if all_queries:
        for q in reversed(all_queries[-5:]):
            preview = q[:34] + "…" if len(q) > 34 else q
            st.markdown(
                f'<div class="history-item">› {preview}</div>', unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div class="history-item" style="color:#1a3a5a;">No queries yet</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
      </div>
      <div>
        <div class="panel-label">Voice Status</div>
        <div class="stat-block" style="font-family:'JetBrains Mono',monospace;font-size:11px;line-height:2;">
          <div style="color:{'#00e676' if VOICE_AVAILABLE else '#aa4050'};">{'◈ Voice Engine Ready' if VOICE_AVAILABLE else '◈ Voice Unavailable'}</div>
          <div style="color:#2a6a8a;">◈ TTS: {'pyttsx3' if VOICE_AVAILABLE else 'Not Installed'}</div>
          <div style="color:#2a6a8a;">◈ STT: {'Google SR' if VOICE_AVAILABLE else 'Not Installed'}</div>
        </div>
      </div>
      <div>
        <div class="panel-label">Data Sources</div>
        <div class="stat-block" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#2a6a8a;line-height:2;">
          <div>◈ Local RAG Engine</div>
          <div>◈ ChromaDB Vector Store</div>
          <div>◈ Llama3 (Ollama)</div>
          <div>◈ MiniLM-L6-v2 Embeddings</div>
          <div>◈ Privacy: Local Only</div>
        </div>
      </div>
      <div>
        <div class="panel-label">Notice</div>
        <div class="disclaimer-box">
          <strong>⚠ Medical Disclaimer</strong>
          For informational reference only. Always consult a licensed healthcare professional before any medical decision.
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
