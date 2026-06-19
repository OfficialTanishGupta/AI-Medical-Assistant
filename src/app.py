import os
import sys
import streamlit as st
import time

# Direct path routing from src folder to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import the RAG engine logic directly
from src.rag import query_rag_engine

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedCore AI — Diagnostic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
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

/* ── Hero Header ── */
.medcore-header {
    background: linear-gradient(135deg, #020d1a 0%, #031a2e 40%, #042338 100%);
    border-bottom: 1px solid rgba(0,210,200,0.2);
    padding: 28px 48px 20px;
    position: relative;
    overflow: hidden;
}

.medcore-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
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
    position: relative;
    z-index: 1;
}

.brand-block {}
.brand-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 3px;
    color: #00c8be;
    text-transform: uppercase;
    margin-bottom: 6px;
    opacity: 0.8;
}
.brand-name {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #e8f4ff;
    line-height: 1;
}
.brand-name span { color: #00d2c8; }
.brand-tagline {
    font-size: 13px;
    font-weight: 300;
    color: #6a96b8;
    margin-top: 6px;
    letter-spacing: 0.3px;
}

/* Vitals strip */
.vitals-strip {
    display: flex;
    gap: 20px;
    align-items: center;
}
.vital-card {
    background: rgba(0,210,200,0.05);
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
    min-width: 90px;
}
.vital-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    color: #4a8aaa;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.vital-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 500;
    color: #00d2c8;
}
.vital-value.green { color: #00e676; }
.vital-value.amber { color: #ffc107; }

/* Pulse line animation */
.pulse-line-container {
    width: 120px;
    height: 40px;
    position: relative;
    overflow: hidden;
}
.pulse-svg { width: 100%; height: 100%; }

/* ── Layout Shell ── */
.main-shell {
    display: grid;
    grid-template-columns: 260px 1fr 260px;
    gap: 0;
    height: calc(100vh - 110px);
    overflow: hidden;
}

/* ── Left Panel ── */
.left-panel {
    background: #020d1a;
    border-right: 1px solid rgba(0,210,200,0.1);
    padding: 24px 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.panel-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    color: #2a5a7a;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,210,200,0.08);
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    background: rgba(0,230,118,0.05);
    border: 1px solid rgba(0,230,118,0.1);
    margin-bottom: 8px;
}
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00e676;
    box-shadow: 0 0 6px #00e676;
    flex-shrink: 0;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #00e676; }
    50% { opacity: 0.5; box-shadow: 0 0 12px #00e676; }
}
.dot.amber { background: #ffc107; box-shadow: 0 0 6px #ffc107; }
.status-text { font-size: 12px; color: #a0c8e0; font-weight: 500; }

.capability-list { display: flex; flex-direction: column; gap: 6px; }
.capability-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid rgba(0,210,200,0.08);
    background: rgba(255,255,255,0.02);
    cursor: default;
    transition: border-color 0.2s, background 0.2s;
}
.capability-item:hover {
    border-color: rgba(0,210,200,0.2);
    background: rgba(0,210,200,0.04);
}
.cap-icon { font-size: 15px; flex-shrink: 0; margin-top: 1px; }
.cap-text {}
.cap-title { font-size: 12px; font-weight: 600; color: #c0dff0; margin-bottom: 2px; }
.cap-desc { font-size: 11px; color: #4a7a9a; line-height: 1.4; }

.quick-queries { display: flex; flex-direction: column; gap: 6px; }
.quick-btn {
    background: rgba(0,210,200,0.04);
    border: 1px solid rgba(0,210,200,0.12);
    border-radius: 7px;
    padding: 9px 12px;
    font-size: 12px;
    color: #7ab4cc;
    cursor: pointer;
    text-align: left;
    font-family: 'Space Grotesk', sans-serif;
    transition: all 0.2s;
    line-height: 1.4;
}
.quick-btn:hover {
    background: rgba(0,210,200,0.09);
    border-color: rgba(0,210,200,0.3);
    color: #b0d8f0;
}

/* ── Center Panel ── */
.center-panel {
    display: flex;
    flex-direction: column;
    background: #030f1e;
    overflow: hidden;
    position: relative;
}

/* Scan line ambient */
.center-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,200,0.4), transparent);
    animation: scan-slide 4s ease-in-out infinite;
    pointer-events: none;
    z-index: 10;
}
@keyframes scan-slide {
    0% { top: 0; opacity: 0; }
    5% { opacity: 1; }
    95% { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.session-header {
    padding: 16px 28px 12px;
    border-bottom: 1px solid rgba(0,210,200,0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
}
.session-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #2a6a8a;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.session-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #1a4a6a;
}

/* Conversation area */
.conv-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    scrollbar-width: thin;
    scrollbar-color: rgba(0,210,200,0.2) transparent;
}
.conv-area::-webkit-scrollbar { width: 4px; }
.conv-area::-webkit-scrollbar-track { background: transparent; }
.conv-area::-webkit-scrollbar-thumb { background: rgba(0,210,200,0.2); border-radius: 2px; }

/* Empty state */
.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 40px;
    text-align: center;
}
.empty-icon { font-size: 48px; opacity: 0.3; }
.empty-title { font-size: 20px; font-weight: 600; color: #3a6a8a; }
.empty-subtitle { font-size: 13px; color: #1a4a6a; max-width: 320px; line-height: 1.6; }

.grid-bg {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,210,200,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,210,200,0.03) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
}

/* Messages */
.msg-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    animation: msg-in 0.3s ease-out;
}
@keyframes msg-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user-row { flex-direction: row-reverse; }

.msg-avatar {
    width: 34px; height: 34px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.msg-avatar.ai-av {
    background: linear-gradient(135deg, rgba(0,210,200,0.15), rgba(0,120,255,0.1));
    border: 1px solid rgba(0,210,200,0.2);
}
.msg-avatar.user-av {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
}

.msg-bubble {
    max-width: 75%;
    padding: 14px 18px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.7;
}
.msg-bubble.ai-bubble {
    background: rgba(0,30,50,0.8);
    border: 1px solid rgba(0,210,200,0.15);
    color: #c0dff0;
    border-top-left-radius: 3px;
}
.msg-bubble.user-bubble {
    background: rgba(0,80,150,0.15);
    border: 1px solid rgba(0,120,255,0.2);
    color: #d0e8ff;
    border-top-right-radius: 3px;
}

.msg-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #2a5a7a;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

/* Source badge */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0,210,200,0.07);
    border: 1px solid rgba(0,210,200,0.15);
    border-radius: 5px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00d2c8;
    margin-top: 10px;
}

/* Input zone */
.input-zone {
    padding: 16px 28px 20px;
    border-top: 1px solid rgba(0,210,200,0.1);
    background: rgba(2,13,26,0.8);
    flex-shrink: 0;
    backdrop-filter: blur(8px);
}

.input-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 2.5px;
    color: #2a5a7a;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Streamlit input overrides */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: rgba(0,20,40,0.6) !important;
    border: 1px solid rgba(0,210,200,0.2) !important;
    border-radius: 10px !important;
    color: #c8e8ff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    padding: 14px 16px !important;
    caret-color: #00d2c8 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: rgba(0,210,200,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,210,200,0.08) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea textarea::placeholder {
    color: #1a4a6a !important;
}
.stTextInput > label, .stTextArea > label {
    display: none !important;
}

/* Submit button */
.stButton > button {
    background: linear-gradient(135deg, #006060, #004a80) !important;
    border: 1px solid rgba(0,210,200,0.35) !important;
    border-radius: 10px !important;
    color: #c8f8f8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 14px 24px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #007070, #005a90) !important;
    border-color: rgba(0,210,200,0.55) !important;
    box-shadow: 0 0 16px rgba(0,210,200,0.15) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Spinner override */
.stSpinner > div { border-top-color: #00d2c8 !important; }

/* ── Right Panel ── */
.right-panel {
    background: #020d1a;
    border-left: 1px solid rgba(0,210,200,0.1);
    padding: 24px 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.stat-block {
    padding: 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(0,210,200,0.08);
    border-radius: 10px;
}
.stat-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: #00d2c8;
    line-height: 1;
    margin: 6px 0 4px;
}
.stat-label { font-size: 11px; color: #4a7a9a; }

.disclaimer-box {
    padding: 14px;
    background: rgba(255,150,0,0.04);
    border: 1px solid rgba(255,150,0,0.12);
    border-radius: 10px;
    font-size: 11px;
    color: #7a6040;
    line-height: 1.6;
}
.disclaimer-box strong { color: #aa8050; display: block; margin-bottom: 4px; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; }

.history-list { display: flex; flex-direction: column; gap: 6px; }
.history-item {
    padding: 9px 12px;
    background: rgba(0,210,200,0.03);
    border: 1px solid rgba(0,210,200,0.07);
    border-radius: 8px;
    font-size: 11px;
    color: #3a6a8a;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Clear button ── */
.clear-btn-area .stButton > button {
    background: rgba(255,50,80,0.06) !important;
    border-color: rgba(255,50,80,0.2) !important;
    color: #aa4050 !important;
    font-size: 12px !important;
    padding: 10px 16px !important;
}
.clear-btn-area .stButton > button:hover {
    background: rgba(255,50,80,0.12) !important;
    box-shadow: none !important;
}

/* ── Responsive adjust ── */
@media (max-width: 1100px) {
    .main-shell { grid-template-columns: 220px 1fr; }
    .right-panel { display: none; }
}
@media (max-width: 800px) {
    .main-shell { grid-template-columns: 1fr; }
    .left-panel { display: none; }
    .medcore-header { padding: 20px 20px 16px; }
    .vitals-strip { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "session_id" not in st.session_state:
    import random, string

    st.session_state.session_id = "MC-" + "".join(
        random.choices(string.ascii_uppercase + string.digits, k=8)
    )


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="medcore-header">
  <div class="header-grid">
    <div class="brand-block">
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
        <div class="vital-value amber" style="font-size:12px;">{st.session_state.session_id}</div>
      </div>
      <div class="vital-card" style="padding:8px 12px;">
        <div class="vital-label">Signal</div>
        <svg class="pulse-svg" viewBox="0 0 120 40" xmlns="http://www.w3.org/2000/svg">
          <polyline points="0,20 20,20 28,8 34,32 40,20 60,20 68,4 74,36 80,20 120,20"
            fill="none" stroke="#00d2c8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.8"/>
        </svg>
      </div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─── Main 3-column layout via Streamlit columns ────────────────────────────────
left_col, center_col, right_col = st.columns([1.1, 3.2, 1.1], gap="small")


# ─── LEFT PANEL ────────────────────────────────────────────────────────────────
with left_col:
    st.markdown(
        """
    <div class="left-panel">
      <div>
        <div class="panel-section-label">System Status</div>
        <div class="status-indicator">
          <div class="dot"></div>
          <span class="status-text">RAG Engine Online</span>
        </div>
        <div class="status-indicator" style="background:rgba(0,120,255,0.05);border-color:rgba(0,120,255,0.15);">
          <div class="dot amber"></div>
          <span class="status-text">Knowledge Base Active</span>
        </div>
      </div>

      <div>
        <div class="panel-section-label">Capabilities</div>
        <div class="capability-list">
          <div class="capability-item">
            <div class="cap-icon">🔬</div>
            <div class="cap-text">
              <div class="cap-title">Symptom Analysis</div>
              <div class="cap-desc">Differential assessment from clinical literature</div>
            </div>
          </div>
          <div class="capability-item">
            <div class="cap-icon">💊</div>
            <div class="cap-text">
              <div class="cap-title">Drug Reference</div>
              <div class="cap-desc">Dosage, interactions & contraindications</div>
            </div>
          </div>
          <div class="capability-item">
            <div class="cap-icon">📋</div>
            <div class="cap-text">
              <div class="cap-title">Clinical Guidelines</div>
              <div class="cap-desc">Evidence-based protocol lookup</div>
            </div>
          </div>
          <div class="capability-item">
            <div class="cap-icon">🧪</div>
            <div class="cap-text">
              <div class="cap-title">Lab Interpretation</div>
              <div class="cap-desc">Reference ranges & result meaning</div>
            </div>
          </div>
        </div>
      </div>

      <div>
        <div class="panel-section-label">Quick Queries</div>
        <div class="quick-queries">
          <div class="quick-btn">What are signs of hypertensive emergency?</div>
          <div class="quick-btn">Normal HbA1c range for diabetics?</div>
          <div class="quick-btn">First-line treatment for UTI?</div>
          <div class="quick-btn">Symptoms of acute appendicitis?</div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ─── CENTER PANEL ──────────────────────────────────────────────────────────────
with center_col:
    st.markdown(
        f"""
    <div class="session-header">
      <span class="session-title">◈ Diagnostic Console</span>
      <span class="session-id">{st.session_state.session_id} · {len(st.session_state.messages)//2} exchanges</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Conversation display
    if not st.session_state.messages:
        st.markdown(
            """
        <div class="empty-state" style="height:340px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;position:relative;">
          <div class="grid-bg"></div>
          <div class="empty-icon">🧬</div>
          <div class="empty-title">Ready for Consultation</div>
          <div class="empty-subtitle">Describe your clinical query, symptoms, or reference question. MedCore AI will retrieve evidence-based answers from your local medical knowledge base.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("time", "--:--")

            if role == "user":
                st.markdown(
                    f"""
                <div class="msg-row user-row">
                  <div class="msg-avatar user-av">👤</div>
                  <div>
                    <div class="msg-meta" style="text-align:right;">Patient Query · {timestamp}</div>
                    <div class="msg-bubble user-bubble">{content}</div>
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="msg-row">
                  <div class="msg-avatar ai-av">🧬</div>
                  <div>
                    <div class="msg-meta">MedCore AI · {timestamp}</div>
                    <div class="msg-bubble ai-bubble">
                      {content}
                      <div class="source-badge">⬡ Retrieved from Local Knowledge Base</div>
                    </div>
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    # Input area
    st.markdown(
        '<div class="input-zone"><div class="input-label">◈ Enter Diagnostic Query</div></div>',
        unsafe_allow_html=True,
    )

    user_question = st.text_input(
        label="query",
        placeholder="Describe symptoms, ask about a drug, or reference a clinical condition…",
        key="query_input",
        label_visibility="collapsed",
    )

    col_submit, col_clear = st.columns([4, 1], gap="small")

    with col_submit:
        submit = st.button("⬡  Run Diagnostic Query", use_container_width=True)

    with col_clear:
        st.markdown('<div class="clear-btn-area">', unsafe_allow_html=True)
        clear = st.button("✕ Clear", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if clear:
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()

    if submit and user_question.strip():
        from datetime import datetime

        ts = datetime.now().strftime("%H:%M")

        st.session_state.messages.append(
            {"role": "user", "content": user_question.strip(), "time": ts}
        )

        with st.spinner("Scanning knowledge base…"):
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


# ─── RIGHT PANEL ───────────────────────────────────────────────────────────────
with right_col:
    st.markdown(
        f"""
    <div class="right-panel">
      <div>
        <div class="panel-section-label">Session Metrics</div>
        <div class="stat-block">
          <div class="stat-label">Total Queries</div>
          <div class="stat-number">{st.session_state.query_count:03d}</div>
        </div>
      </div>

      <div>
        <div class="panel-section-label">Recent Topics</div>
        <div class="history-list">
    """,
        unsafe_allow_html=True,
    )

    user_msgs = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        for q in reversed(user_msgs[-5:]):
            preview = q[:38] + "…" if len(q) > 38 else q
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
      </div>

      <div>
        <div class="panel-section-label">Notice</div>
        <div class="disclaimer-box">
          <strong>⚠ Medical Disclaimer</strong>
          This system references clinical guidelines for informational purposes only. Always consult a licensed healthcare professional before making any medical decisions.
        </div>
      </div>

      <div>
        <div class="panel-section-label">Data Source</div>
        <div class="stat-block" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#2a6a8a;line-height:1.8;">
          <div>◈ Local RAG Engine</div>
          <div>◈ Vector Store: Active</div>
          <div>◈ Embeddings: Online</div>
          <div>◈ Privacy: Local Only</div>
        </div>
      </div>

    </div>
    """,
        unsafe_allow_html=True,
    )
