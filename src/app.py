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

st.set_page_config(
    page_title="MedCore AI — Diagnostic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def search_medicine_details(med_name: str):
    """Searches the vector database for a specific medicine and extracts structured details."""
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

        retrieved_chunks = []
        if results and "documents" in results and results["documents"]:
            # ChromaDB returns a list of lists; extract the inner list safely
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


def show_loader(label: str = "Processing", sublabel: str = "Accessing knowledge base…"):
    """Injects a full-screen futuristic loading overlay via HTML/JS."""
    st.markdown(
        f"""
    <style>
    #medcore-loader {{
        position: fixed; inset: 0; z-index: 99999;
        background: rgba(2, 13, 26, 0.92);
        backdrop-filter: blur(6px);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        gap: 24px; pointer-events: all;
        animation: loader-in 0.2s ease-out;
    }}
    @keyframes loader-in {{ from{{opacity:0}} to{{opacity:1}} }}

    .ldr-ring-wrap {{ position: relative; width: 90px; height: 90px; }}
    .ldr-ring {{
        position: absolute; inset: 0;
        border-radius: 50%;
        border: 2px solid transparent;
    }}
    .ldr-ring.r1 {{
        border-top-color: #00d2c8;
        animation: spin 1.1s linear infinite;
    }}
    .ldr-ring.r2 {{
        inset: 10px;
        border-right-color: rgba(0,210,200,0.4);
        animation: spin 0.75s linear infinite reverse;
    }}
    .ldr-ring.r3 {{
        inset: 20px;
        border-bottom-color: rgba(0,210,200,0.2);
        animation: spin 1.6s linear infinite;
    }}
    .ldr-core {{
        position: absolute; inset: 32px;
        background: radial-gradient(circle, rgba(0,210,200,0.25), transparent);
        border-radius: 50%;
        animation: core-pulse 1.2s ease-in-out infinite;
    }}
    @keyframes spin {{ to{{ transform: rotate(360deg); }} }}
    @keyframes core-pulse {{ 0%,100%{{opacity:0.4;transform:scale(0.9)}} 50%{{opacity:1;transform:scale(1.1)}} }}

    .ldr-text-block {{ text-align: center; }}
    .ldr-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px; font-weight: 500;
        letter-spacing: 3px; text-transform: uppercase;
        color: #00d2c8; margin-bottom: 6px;
    }}
    .ldr-sublabel {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 13px; color: #3a7a9a;
    }}

    .ldr-bar-track {{
        width: 260px; height: 2px;
        background: rgba(0,210,200,0.1);
        border-radius: 2px; overflow: hidden;
    }}
    .ldr-bar-fill {{
        height: 100%; width: 40%;
        background: linear-gradient(90deg, transparent, #00d2c8, transparent);
        border-radius: 2px;
        animation: bar-sweep 1.4s ease-in-out infinite;
    }}
    @keyframes bar-sweep {{
        0%   {{ transform: translateX(-200%); }}
        100% {{ transform: translateX(500%); }}
    }}

    .ldr-dots {{
        display: flex; gap: 8px;
    }}
    .ldr-dot {{
        width: 5px; height: 5px; border-radius: 50%;
        background: rgba(0,210,200,0.3);
        animation: dot-pop 1.2s ease-in-out infinite;
    }}
    .ldr-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .ldr-dot:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes dot-pop {{ 0%,80%,100%{{transform:scale(1);opacity:0.3}} 40%{{transform:scale(1.6);opacity:1;background:#00d2c8}} }}
    </style>

    <div id="medcore-loader">
      <div class="ldr-ring-wrap">
        <div class="ldr-ring r1"></div>
        <div class="ldr-ring r2"></div>
        <div class="ldr-ring r3"></div>
        <div class="ldr-core"></div>
      </div>
      <div class="ldr-text-block">
        <div class="ldr-label">{label}</div>
        <div class="ldr-sublabel">{sublabel}</div>
      </div>
      <div class="ldr-bar-track"><div class="ldr-bar-fill"></div></div>
      <div class="ldr-dots">
        <div class="ldr-dot"></div>
        <div class="ldr-dot"></div>
        <div class="ldr-dot"></div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #020d1a !important;
    color: #c8dff0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stAppViewContainer"] { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Header ── */
.medcore-header {
    background: linear-gradient(135deg, #020d1a 0%, #031a2e 40%, #042338 100%);
    border-bottom: 1px solid rgba(0,210,200,0.2);
    padding: 24px 48px 18px;
    position: relative;
    overflow: hidden;
}
.medcore-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 60% 80% at 85% 50%, rgba(0,180,200,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 30% 60% at 10% 50%, rgba(0,120,255,0.05) 0%, transparent 70%);
    pointer-events: none;
}
.header-grid {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 24px;
    position: relative; z-index: 1;
}
.brand-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 400;
    letter-spacing: 3px; color: #00c8be;
    text-transform: uppercase; margin-bottom: 6px; opacity: 0.8;
}
.brand-name { font-size: 30px; font-weight: 700; letter-spacing: -0.5px; color: #e8f4ff; line-height: 1; }
.brand-name span { color: #00d2c8; }
.brand-tagline { font-size: 12px; font-weight: 300; color: #6a96b8; margin-top: 5px; letter-spacing: 0.3px; }
.vitals-strip { display: flex; gap: 16px; align-items: center; }
.vital-card {
    background: rgba(0,210,200,0.05);
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 8px; padding: 9px 14px; text-align: center; min-width: 84px;
}
.vital-label { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 2px; color: #4a8aaa; text-transform: uppercase; margin-bottom: 4px; }
.vital-value { font-family: 'JetBrains Mono', monospace; font-size: 17px; font-weight: 500; color: #00d2c8; }
.vital-value.green { color: #00e676; }
.vital-value.amber { color: #ffc107; font-size: 12px; }

/* mode-bar handled via st.columns + .mode-btn-wrap above */

/* ── Layout ── */
.main-shell { display: grid; grid-template-columns: 240px 1fr 240px; height: calc(100vh - 116px); overflow: hidden; }

/* ── Left Panel ── */
.left-panel {
    background: #020d1a;
    border-right: 1px solid rgba(0,210,200,0.1);
    padding: 22px 18px;
    overflow-y: auto;
    display: flex; flex-direction: column; gap: 20px;
}
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 3px; color: #2a5a7a;
    text-transform: uppercase; margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,210,200,0.08);
}
.status-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 11px; border-radius: 8px;
    background: rgba(0,230,118,0.05);
    border: 1px solid rgba(0,230,118,0.1);
    margin-bottom: 7px;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: #00e676; box-shadow: 0 0 6px #00e676; flex-shrink: 0; animation: pulse-dot 2s ease-in-out infinite; }
.dot.amber { background: #ffc107; box-shadow: 0 0 6px #ffc107; }
@keyframes pulse-dot { 0%,100%{opacity:1;box-shadow:0 0 6px #00e676;} 50%{opacity:0.5;box-shadow:0 0 12px #00e676;} }
.status-text { font-size: 12px; color: #a0c8e0; font-weight: 500; }

.cap-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 11px; border-radius: 8px;
    border: 1px solid rgba(0,210,200,0.08);
    background: rgba(255,255,255,0.02);
    margin-bottom: 6px;
    transition: border-color 0.2s, background 0.2s;
}
.cap-item:hover { border-color: rgba(0,210,200,0.2); background: rgba(0,210,200,0.04); }
.cap-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
.cap-title { font-size: 12px; font-weight: 600; color: #c0dff0; margin-bottom: 2px; }
.cap-desc { font-size: 11px; color: #4a7a9a; line-height: 1.4; }

.quick-btn {
    display: block; width: 100%;
    background: rgba(0,210,200,0.04);
    border: 1px solid rgba(0,210,200,0.12);
    border-radius: 7px; padding: 9px 11px;
    font-size: 12px; color: #7ab4cc;
    cursor: pointer; text-align: left;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.2s; line-height: 1.4; margin-bottom: 5px;
}
.quick-btn:hover { background: rgba(0,210,200,0.09); border-color: rgba(0,210,200,0.3); color: #b0d8f0; }

/* ── Center Panel ── */
.center-panel { display: flex; flex-direction: column; background: #030f1e; overflow: hidden; position: relative; }
.center-panel::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,200,0.4), transparent);
    animation: scan-slide 5s ease-in-out infinite; pointer-events: none; z-index: 10;
}
@keyframes scan-slide { 0%{top:0;opacity:0} 5%{opacity:1} 95%{opacity:1} 100%{top:100%;opacity:0} }

.console-header {
    padding: 14px 26px;
    border-bottom: 1px solid rgba(0,210,200,0.08);
    display: flex; align-items: center; justify-content: space-between;
    flex-shrink: 0;
}
.console-title { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #2a6a8a; letter-spacing: 2px; text-transform: uppercase; }
.console-meta { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #1a4a6a; }

/* Messages */
.conv-area {
    flex: 1; overflow-y: auto; padding: 22px 26px;
    display: flex; flex-direction: column; gap: 18px;
    scrollbar-width: thin; scrollbar-color: rgba(0,210,200,0.2) transparent;
}
.conv-area::-webkit-scrollbar { width: 4px; }
.conv-area::-webkit-scrollbar-thumb { background: rgba(0,210,200,0.2); border-radius: 2px; }

.empty-state {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 12px; padding: 40px; text-align: center;
    min-height: 300px; position: relative;
}
.grid-bg {
    position: absolute; inset: 0;
    background-image: linear-gradient(rgba(0,210,200,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,210,200,0.03) 1px, transparent 1px);
    background-size: 32px 32px; pointer-events: none;
}
.empty-icon { font-size: 44px; opacity: 0.25; }
.empty-title { font-size: 18px; font-weight: 600; color: #3a6a8a; }
.empty-sub { font-size: 13px; color: #1a4a6a; max-width: 300px; line-height: 1.6; }

.msg-row { display: flex; gap: 12px; align-items: flex-start; animation: msg-in 0.3s ease-out; }
@keyframes msg-in { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.msg-row.user-row { flex-direction: row-reverse; }
.msg-avatar {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0;
}
.msg-avatar.ai-av { background: linear-gradient(135deg, rgba(0,210,200,0.15), rgba(0,120,255,0.1)); border: 1px solid rgba(0,210,200,0.2); }
.msg-avatar.user-av { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); }
.msg-meta { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #2a5a7a; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 5px; }
.msg-bubble { max-width: 76%; padding: 13px 16px; border-radius: 12px; font-size: 14px; line-height: 1.7; }
.msg-bubble.ai { background: rgba(0,30,50,0.8); border: 1px solid rgba(0,210,200,0.15); color: #c0dff0; border-top-left-radius: 3px; }
.msg-bubble.user { background: rgba(0,80,150,0.15); border: 1px solid rgba(0,120,255,0.2); color: #d0e8ff; border-top-right-radius: 3px; }
.src-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(0,210,200,0.07); border: 1px solid rgba(0,210,200,0.15);
    border-radius: 5px; padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #00d2c8; margin-top: 10px;
}

/* ── Medicine Search Panel ── */
.med-search-area {
    flex: 1; overflow-y: auto; padding: 28px 30px;
    scrollbar-width: thin; scrollbar-color: rgba(0,210,200,0.2) transparent;
}
.med-search-area::-webkit-scrollbar { width: 4px; }
.med-search-area::-webkit-scrollbar-thumb { background: rgba(0,210,200,0.2); }

.med-intro {
    margin-bottom: 20px;
    padding: 16px 20px;
    background: rgba(0,210,200,0.04);
    border: 1px solid rgba(0,210,200,0.1);
    border-radius: 10px;
    font-size: 13px; color: #5a9ab8; line-height: 1.6;
}
.med-intro strong { color: #00d2c8; display: block; margin-bottom: 4px; font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; }

.med-result-card {
    background: rgba(0,20,40,0.6);
    border: 1px solid rgba(0,210,200,0.18);
    border-radius: 12px;
    padding: 24px 26px;
    margin-top: 20px;
}
.med-result-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; letter-spacing: 2px;
    text-transform: uppercase; color: #00d2c8;
    margin-bottom: 16px; padding-bottom: 12px;
    border-bottom: 1px solid rgba(0,210,200,0.12);
    display: flex; align-items: center; gap: 10px;
}
.med-result-body { font-size: 14px; color: #b0d0e8; line-height: 1.8; }
.med-result-body h3 { font-size: 13px; color: #00d2c8; margin: 18px 0 8px; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px; }
.med-result-body p { margin-bottom: 10px; }

/* ── Input Zone ── */
.input-zone {
    padding: 14px 26px 18px;
    border-top: 1px solid rgba(0,210,200,0.1);
    background: rgba(2,13,26,0.85);
    flex-shrink: 0;
    backdrop-filter: blur(8px);
}
.input-zone-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.5px; color: #2a5a7a;
    text-transform: uppercase; margin-bottom: 8px;
}

/* Streamlit input overrides */
.stTextInput > div > div > input {
    background: rgba(0,20,40,0.6) !important;
    border: 1px solid rgba(0,210,200,0.2) !important;
    border-radius: 10px !important;
    color: #c8e8ff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    padding: 13px 16px !important;
    caret-color: #00d2c8 !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(0,210,200,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,210,200,0.08) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #1a4a6a !important; }
.stTextInput > label { display: none !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #006060, #004a80) !important;
    border: 1px solid rgba(0,210,200,0.35) !important;
    border-radius: 10px !important;
    color: #c8f8f8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 13px 22px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #007070, #005a90) !important;
    border-color: rgba(0,210,200,0.55) !important;
    box-shadow: 0 0 16px rgba(0,210,200,0.15) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.clear-wrap .stButton > button {
    background: rgba(255,50,80,0.06) !important;
    border-color: rgba(255,50,80,0.2) !important;
    color: #aa4050 !important;
    font-size: 12px !important; padding: 13px 14px !important;
    box-shadow: none !important;
}
.clear-wrap .stButton > button:hover {
    background: rgba(255,50,80,0.12) !important;
    box-shadow: none !important; transform: none !important;
}

/* ── Right Panel ── */
.right-panel {
    background: #020d1a;
    border-left: 1px solid rgba(0,210,200,0.1);
    padding: 22px 18px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 18px;
}
.stat-block {
    padding: 14px 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0,210,200,0.08);
    border-radius: 10px;
}
.stat-number { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 500; color: #00d2c8; line-height: 1; margin: 5px 0 3px; }
.stat-label { font-size: 11px; color: #4a7a9a; }
.disclaimer-box {
    padding: 13px 14px;
    background: rgba(255,150,0,0.04);
    border: 1px solid rgba(255,150,0,0.12);
    border-radius: 10px;
    font-size: 11px; color: #7a6040; line-height: 1.6;
}
.disclaimer-box strong { color: #aa8050; display: block; margin-bottom: 4px; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }
.history-item {
    padding: 8px 11px;
    background: rgba(0,210,200,0.03);
    border: 1px solid rgba(0,210,200,0.07);
    border-radius: 8px;
    font-size: 11px; color: #3a6a8a;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    margin-bottom: 5px;
}

/* Streamlit tab override — hide them visually, we handle mode ourselves */
[data-testid="stTabs"] { display: none !important; }

@media (max-width: 1100px) { .main-shell { grid-template-columns: 200px 1fr; } .right-panel { display: none; } }
@media (max-width: 800px) { .main-shell { grid-template-columns: 1fr; } .left-panel { display: none; } .vitals-strip { display: none; } .medcore-header { padding: 18px 20px 14px; } }
</style>
""",
    unsafe_allow_html=True,
)


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
      <div class="vital-card">
        <div class="vital-label">RAG Engine</div>
        <div class="vital-value green">LIVE</div>
      </div>
      <div class="vital-card">
        <div class="vital-label">Queries</div>
        <div class="vital-value">{st.session_state.query_count:03d}</div>
      </div>
      <div class="vital-card">
        <div class="vital-label">Session</div>
        <div class="vital-value amber">{st.session_state.session_id}</div>
      </div>
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

st.markdown(
    """
<style>
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
    background: #020d1a;
    border-bottom: 1px solid rgba(0,210,200,0.1);
    padding: 0 48px;
    gap: 0 !important;
}
/* Mode tab button base */
div[data-testid="column"]:has(.mode-btn-wrap) {
    flex: 0 0 auto !important;
    width: auto !important;
    padding: 0 !important;
}
.mode-btn-wrap .stButton > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #2a5a7a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 13px 28px !important;
    width: auto !important;
    box-shadow: none !important;
    transition: color 0.2s !important;
}
.mode-btn-wrap .stButton > button:hover {
    color: #6aaac8 !important;
    background: transparent !important;
    box-shadow: none !important;
    transform: none !important;
    border-bottom: 2px solid transparent !important;
}
.mode-btn-wrap.active .stButton > button {
    color: #00d2c8 !important;
    border-bottom: 2px solid #00d2c8 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

mode_col1, mode_col2, mode_spacer = st.columns([1, 1, 8], gap="small")
with mode_col1:
    chat_active_cls = "active" if st.session_state.active_mode == "chat" else ""
    st.markdown(
        f'<div class="mode-btn-wrap {chat_active_cls}">', unsafe_allow_html=True
    )
    if st.button("◈  Clinical Chat AI", key="mode_chat"):
        st.session_state.active_mode = "chat"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with mode_col2:
    med_active_cls = "active" if st.session_state.active_mode == "med" else ""
    st.markdown(f'<div class="mode-btn-wrap {med_active_cls}">', unsafe_allow_html=True)
    if st.button("⬡  Medicine Search", key="mode_med"):
        st.session_state.active_mode = "med"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<style>
.vitals-graph-bar {
    background: linear-gradient(90deg, #020d1a, #031625, #020d1a);
    border-bottom: 1px solid rgba(0,210,200,0.12);
    padding: 10px 48px;
    display: flex;
    align-items: center;
    gap: 16px;
    overflow: hidden;
    position: relative;
}
.vgb-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.5px; color: #1a5a7a;
    text-transform: uppercase; white-space: nowrap; flex-shrink: 0;
}
.vgb-canvas-wrap { flex: 1; height: 48px; position: relative; }
canvas.vitals-canvas { width: 100%; height: 48px; display: block; }
.vgb-legend {
    display: flex; gap: 14px; flex-shrink: 0;
}
.vgb-leg-item {
    display: flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #3a7a9a;
}
.vgb-leg-dot { width: 8px; height: 2px; border-radius: 1px; }
</style>

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
(function() {
  const canvas = document.getElementById('vitalsChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth  * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }
  resize();

  const W = () => canvas.offsetWidth;
  const H = () => canvas.offsetHeight;
  const POINTS = 120;

  // Seed data
  function makeWave(base, amp, freq, noise) {
    return Array.from({length: POINTS}, (_, i) =>
      base + amp * Math.sin(i * freq) + (Math.random() - 0.5) * noise
    );
  }

  let hr   = makeWave(0.5, 0.28, 0.18, 0.06);
  let spo2 = makeWave(0.75, 0.12, 0.09, 0.03);
  let bp   = makeWave(0.35, 0.20, 0.13, 0.05);

  function pushPoint(arr, base, amp, freq, noise, t) {
    arr.push(base + amp * Math.sin(t * freq) + (Math.random() - 0.5) * noise);
    if (arr.length > POINTS) arr.shift();
  }

  let t = POINTS;

  function drawLine(data, color, lineW) {
    const w = W(), h = H();
    const step = w / (POINTS - 1);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.lineJoin = 'round';
    ctx.lineCap  = 'round';
    data.forEach((v, i) => {
      const x = i * step;
      const y = h - v * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Glow pass
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW + 3;
    ctx.globalAlpha = 0.12;
    data.forEach((v, i) => {
      const x = i * step;
      const y = h - v * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1.0;
  }

  function drawScanLine() {
    // Animated vertical scan cursor at the tip
    const w = W(), h = H();
    const x = w - 1;
    const grad = ctx.createLinearGradient(x - 20, 0, x, 0);
    grad.addColorStop(0, 'rgba(0,210,200,0)');
    grad.addColorStop(1, 'rgba(0,210,200,0.25)');
    ctx.fillStyle = grad;
    ctx.fillRect(x - 20, 0, 20, h);
  }

  function frame() {
    const w = W(), h = H();
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(0,210,200,0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (i / 4) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    for (let i = 0; i <= 8; i++) {
      const x = (i / 8) * w;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    drawScanLine();
    drawLine(spo2, '#00e676', 1.2);
    drawLine(bp,   '#ffc107', 1.2);
    drawLine(hr,   '#00d2c8', 1.8);

    // Advance
    pushPoint(hr,   0.5,  0.28, 0.18, 0.06, t);
    pushPoint(spo2, 0.75, 0.12, 0.09, 0.03, t);
    pushPoint(bp,   0.35, 0.20, 0.13, 0.05, t);
    t++;

    requestAnimationFrame(frame);
  }

  window.addEventListener('resize', resize);
  frame();
})();
</script>
""",
    unsafe_allow_html=True,
)

left_col, center_col, right_col = st.columns([1.0, 3.4, 1.0], gap="small")

with left_col:
    st.markdown(
        """
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
      </div>
      <div>
        <div class="panel-label">Capabilities</div>
        <div class="cap-item"><div class="cap-icon">🔬</div><div><div class="cap-title">Symptom Analysis</div><div class="cap-desc">Differential assessment from clinical literature</div></div></div>
        <div class="cap-item"><div class="cap-icon">💊</div><div><div class="cap-title">Drug Reference</div><div class="cap-desc">Dosage, interactions & contraindications</div></div></div>
        <div class="cap-item"><div class="cap-icon">📋</div><div><div class="cap-title">Clinical Guidelines</div><div class="cap-desc">Evidence-based protocol lookup</div></div></div>
        <div class="cap-item"><div class="cap-icon">🧪</div><div><div class="cap-title">Lab Interpretation</div><div class="cap-desc">Reference ranges & result meaning</div></div></div>
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


with center_col:

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

        if not st.session_state.messages:
            st.markdown(
                """
            <div class="empty-state">
              <div class="grid-bg"></div>
              <div class="empty-icon">🧬</div>
              <div class="empty-title">Ready for Consultation</div>
              <div class="empty-sub">Describe symptoms, ask about a condition, or reference clinical guidelines. MedCore AI retrieves evidence-based answers from your local knowledge base.</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                ts = msg.get("time", "--:--")
                if role == "user":
                    st.markdown(
                        f"""
                    <div class="msg-row user-row">
                      <div class="msg-avatar user-av">👤</div>
                      <div>
                        <div class="msg-meta" style="text-align:right;">Patient Query · {ts}</div>
                        <div class="msg-bubble user">{content}</div>
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
                        <div class="msg-bubble ai">{content}<div class="src-badge">⬡ Retrieved from Local Knowledge Base</div></div>
                      </div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

        # Input
        st.markdown(
            '<div class="input-zone"><div class="input-zone-label">◈ Enter Diagnostic Query</div></div>',
            unsafe_allow_html=True,
        )

        user_question = st.text_input(
            "query",
            label_visibility="collapsed",
            placeholder="Describe symptoms, ask about a drug, or reference a clinical condition…",
            key="chat_input",
        )

        c1, c2 = st.columns([4, 1], gap="small")
        with c1:
            submit = st.button(
                "⬡  Run Diagnostic Query", use_container_width=True, key="chat_submit"
            )
        with c2:
            st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
            clear = st.button("✕ Clear", use_container_width=True, key="chat_clear")
            st.markdown("</div>", unsafe_allow_html=True)

        if clear:
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()

        if submit and user_question.strip():
            ts = datetime.now().strftime("%H:%M")
            st.session_state.messages.append(
                {"role": "user", "content": user_question.strip(), "time": ts}
            )

            show_loader("Running Diagnostic Query", "Scanning RAG knowledge base…")
            try:
                answer = query_rag_engine(user_question.strip())
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
            st.rerun()

    # ── MODE: MEDICINE SEARCH ────────────────────────────────────────────────
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
        <div class="med-search-area">
          <div class="med-intro">
            <strong>◈ Knowledge Base Lookup</strong>
            Enter a generic or brand medicine name to retrieve its structured clinical profile — uses, side effects, warnings, and dosage — directly from your local vector database.
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Input
        st.markdown(
            '<div class="input-zone"><div class="input-zone-label">⬡ Medicine Name</div></div>',
            unsafe_allow_html=True,
        )

        med_query = st.text_input(
            "med",
            label_visibility="collapsed",
            placeholder="e.g. Paracetamol, Metformin, Aconitum Napellus…",
            key="med_input",
        )

        med_submit = st.button(
            "⬡  Retrieve Clinical Profile", use_container_width=True, key="med_submit"
        )
        st.markdown(
            '<div class="clear-wrap" style="margin-top:6px;">', unsafe_allow_html=True
        )
        med_clear = st.button(
            "✕ Clear Results", use_container_width=True, key="med_clear"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if med_clear:
            st.session_state.med_result = None
            st.session_state.med_name_searched = ""
            st.rerun()

        if med_submit and med_query.strip():
            show_loader(
                "Retrieving Clinical Profile", f"Querying vector store for {med_query}…"
            )
            result = search_medicine_details(med_query.strip())
            st.session_state.med_result = result
            st.session_state.med_name_searched = med_query.strip()
            st.session_state.query_count += 1
            st.rerun()

        # Show result
        if st.session_state.med_result:
            st.markdown(
                f"""
            <div class="med-result-card">
              <div class="med-result-title">
                💊 Clinical Profile — {st.session_state.med_name_searched.upper()}
              </div>
              <div class="med-result-body">
                {st.session_state.med_result.replace(chr(10), '<br>')}
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


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
        """
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
