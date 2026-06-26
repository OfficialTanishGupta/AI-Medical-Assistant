import os
import sys
import streamlit as st
from datetime import datetime
import random
import string
import speech_recognition as sr
import pyttsx3
from pypdf import PdfReader

# Direct path routing from src folder to project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.rag import query_rag_engine
import ollama
import chromadb
from chromadb.utils import embedding_functions

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedCore AI — Diagnostic Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── Voice Engine Initialization ──────────────────────────────────────────────
def speak_text(text: str):
    """Converts text blocks to audible speech synthesis offline."""
    try:
        clean_text = text.replace("*", "").replace("#", "").replace("-", " ")
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.say(clean_text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Text-to-Speech Error: {str(e)}")


def listen_to_microphone():
    """Captures microphone audio and returns transcribed text string."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.toast("🎙️ Listening... Speak your medical question now.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            st.toast("✨ Processing audio recording...")
            return recognizer.recognize_google(audio)
        except Exception as e:
            st.error(f"[!] Audio input error: {str(e)}")
    return ""


# ─── Medicine Search Helper ────────────────────────────────────────────────────
def search_medicine_details(med_name: str):
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
        raw_docs = results["documents"] if results and results["documents"] else []

        # Flatten nested lists safely
        flat_chunks = []
        for item in raw_docs:
            if isinstance(item, list):
                flat_chunks.extend(item)
            else:
                flat_chunks.append(item)
        context_str = "\n\n".join(flat_chunks)

        system_prompt = (
            f"You are a clinical pharmacology assistant. Analyze the reference material regarding '{med_name}'. "
            "Provide a clear, clinical summary strictly structured into three exact headings:\n"
            "### 📋 Uses\n"
            "### ⚠️ Side Effects & Warnings\n"
            "### 💊 Dosage Information\n\n"
            "If the reference material does not contain specific information for a section, use your "
            "comprehensive internal medical knowledge to fill in standard clinical guidelines for that medicine."
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


# ─── Medical Report Parsing & Summarization Engine ────────────────────────────
def summarize_medical_report(file_stream):
    """Parses raw text layers out of an uploaded PDF report and structures analysis."""
    try:
        reader = PdfReader(file_stream)
        raw_text = ""
        for page in reader.pages:
            text_content = page.extract_text()
            if text_content:
                raw_text += text_content + "\n"

        if not raw_text.strip():
            return (
                "❌ Could not extract any readable text layers from the uploaded file. "
                "Ensure it is not a raw flat image scan."
            )

        system_prompt = (
            "You are an advanced clinical diagnostics officer. Analyze the text extracted from this medical report. "
            "Generate a highly structured, objective summary using exactly these four primary clinical headings:\n"
            "### 📋 Document Summary\n"
            "(Identify the document type, date, patient profile markers, and test provider details smoothly)\n\n"
            "### 🔍 Key Biomarker Values\n"
            "(List significant laboratory indicators, normal reference boundaries, out-of-range metrics, or diagnostic line items cleanly)\n\n"
            "### 💡 Clinical Observations\n"
            "(Explain the biological implications of the findings in easy-to-understand terms without sounding robotic)\n\n"
            "### 🎯 Next-Step Recommendations\n"
            "(Outline further diagnostic screening profiles, behavioral considerations, or specific consultations required safely)"
        )

        response = ollama.generate(
            model="llama3",
            system=system_prompt,
            prompt=f"Extracted Report Content Text:\n{raw_text}",
            options={"temperature": 0.2},
        )
        return response["response"]
    except Exception as e:
        return f"[!] Analysis pipeline failed: {str(e)}"


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
if "voice_input_buffer" not in st.session_state:
    st.session_state.voice_input_buffer = ""
if "report_analysis_output" not in st.session_state:
    st.session_state.report_analysis_output = None


# ─── CSS Theme Override ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    *, *::before, *::after { box-sizing: border-box; margin: 0; }
    html, body, [data-testid="stAppViewContainer"] {
        background: #020d1a !important;
        color: #c8dff0 !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    footer { display: none !important; }
    .medcore-header {
        background: linear-gradient(135deg, #020d1a 0%, #031a2e 40%, #042338 100%);
        border-bottom: 1px solid rgba(0,210,200,0.2);
        padding: 24px 48px;
    }
    .header-grid { display: flex; justify-content: space-between; align-items: center; }
    .brand-name { font-size: 30px; font-weight: 700; color: #e8f4ff; }
    .brand-name span { color: #00d2c8; }
    .status-box {
        background: rgba(0, 210, 200, 0.04);
        border: 1px solid rgba(0, 210, 200, 0.15);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        color: #00d2c8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Custom UI Header ──────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="medcore-header">
        <div class="header-grid">
            <div>
                <div class="brand-name">MedCore <span>AI</span></div>
                <div style="color: #6a96b8; font-size: 13px; margin-top: 4px;">
                    Local Diagnostic Support &amp; Clinical RAG Engine
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 10px; color: #4a8aaa; font-family: 'JetBrains Mono';">SESSION ID</div>
                <div style="color: #00d2c8; font-family: 'JetBrains Mono'; font-weight: 500;">
                    {st.session_state.session_id}
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ─── Layout Split ──────────────────────────────────────────────────────────────
col_left, col_main, col_right = st.columns([1, 3, 1])

# ─── Left Sidebar Panel: Status & Capabilities ────────────────────────────────
with col_left:
    st.markdown("### SYSTEM STATUS")
    st.markdown(
        """
        <div class="status-box">
            <span style="color: #00e676;">●</span> <b>RAG Engine Online</b><br>
            <span style="color: #00e676;">●</span> <b>Knowledge Base Active</b><br>
            <span style="color: #00e676;">●</span> <b>Report Engine Active</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### CAPABILITIES")
    st.info(
        "📊 **Report Summarization**\n\nExtracts biomarkers, metrics, and observations from lab PDFs."
    )

# ─── Right Panel: Session Statistics & Metrics ────────────────────────────────
with col_right:
    st.markdown("### SESSION METRICS")
    st.markdown(
        f'<div class="status-box">'
        f'<div style="font-size: 11px; color: #4a8aaa;">TOTAL QUERIES</div>'
        f'<div class="metric-value">{st.session_state.query_count:03d}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### VOICE INTERACTION")
    if st.button("🎙️ Speak Question", use_container_width=True, type="primary"):
        spoken_text = listen_to_microphone()
        if spoken_text:
            st.session_state.voice_input_buffer = spoken_text
            st.session_state.active_mode = "chat"
            st.rerun()

# ─── Central Main Application Workspace Panel ─────────────────────────────────
with col_main:
    # Mode selector tabs
    mode_index = 0
    if st.session_state.active_mode == "medicine":
        mode_index = 1
    elif st.session_state.active_mode == "report":
        mode_index = 2

    mode_selection = st.radio(
        "Select Active Workspace Mode:",
        ["💬 Clinical Chat AI", "🔍 Medicine Search", "📄 Report Summarizer"],
        index=mode_index,
        horizontal=True,
    )
    st.markdown("---")

    # ── WORKSPACE MODE 1: CLINICAL CHAT AI ────────────────────────────────────
    if mode_selection == "💬 Clinical Chat AI":
        st.session_state.active_mode = "chat"
        st.subheader("Clinical Consultation Terminal")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        query_to_process = ""
        if st.session_state.voice_input_buffer:
            query_to_process = st.session_state.voice_input_buffer
            st.session_state.voice_input_buffer = ""
        else:
            user_type_input = st.chat_input(
                "Enter symptoms or therapeutic questions..."
            )
            if user_type_input:
                query_to_process = user_type_input

        if query_to_process:
            with st.chat_message("user"):
                st.markdown(query_to_process)
            st.session_state.messages.append(
                {"role": "user", "content": query_to_process}
            )
            st.session_state.query_count += 1

            with st.chat_message("assistant"):
                with st.spinner("Reviewing medical reference documentation..."):
                    response = query_rag_engine(query_to_process)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            speak_text(response)
            st.rerun()

    # ── WORKSPACE MODE 2: MEDICINE SEARCH ─────────────────────────────────────
    elif mode_selection == "🔍 Medicine Search":
        st.session_state.active_mode = "medicine"
        st.subheader("Medicine Search — Clinical Profile Retrieval")

        med_input = st.text_input(
            "MEDICINE NAME",
            value=st.session_state.med_name_searched,
            placeholder="e.g., Paracetamol, Arnica Montana",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(
                "○ Retrieve Clinical Profile", use_container_width=True, type="primary"
            ):
                if med_input.strip():
                    st.session_state.med_name_searched = med_input.strip()
                    st.session_state.query_count += 1
                    with st.spinner(f"Extracting clinical files for {med_input}..."):
                        st.session_state.med_result = search_medicine_details(
                            med_input.strip()
                        )
                else:
                    st.warning("Please specify a valid medicine parameter.")

        with col_btn2:
            if st.button("✕ Clear Results", use_container_width=True):
                st.session_state.med_result = None
                st.session_state.med_name_searched = ""
                st.rerun()

        if st.session_state.med_result:
            st.markdown(
                f"### 🧬 CLINICAL PROFILE — {st.session_state.med_name_searched.upper()}"
            )
            st.markdown(st.session_state.med_result)

    # ── WORKSPACE MODE 3: MEDICAL REPORT SUMMARIZER ───────────────────────────
    elif mode_selection == "📄 Report Summarizer":
        st.session_state.active_mode = "report"
        st.subheader("Diagnostic Report Processing Module")
        st.markdown(
            "Upload digital lab sheets, blood panels, or prescriptions to run an evaluation."
        )

        uploaded_file = st.file_uploader(
            "Upload Report File (PDF Format Only)", type=["pdf"]
        )

        col_rep1, col_rep2 = st.columns(2)
        with col_rep1:
            if st.button(
                "🚀 Process & Summarize Report",
                use_container_width=True,
                type="primary",
            ):
                if uploaded_file is not None:
                    st.session_state.query_count += 1
                    with st.spinner(
                        "Parsing report values and evaluating biomarkers locally..."
                    ):
                        st.session_state.report_analysis_output = (
                            summarize_medical_report(uploaded_file)
                        )
                else:
                    st.warning("Please upload a valid medical report file to proceed.")

        with col_rep2:
            if st.button("✕ Clear Report Data", use_container_width=True):
                st.session_state.report_analysis_output = None
                st.rerun()

        if st.session_state.report_analysis_output:
            st.markdown("---")
            st.markdown("## 📊 LOCAL DIAGNOSTIC REPORT EVALUATION")
            st.markdown(st.session_state.report_analysis_output)
            if st.button("🔊 Read Evaluation Aloud", use_container_width=True):
                speak_text(st.session_state.report_analysis_output)


# ─── Mandatory Health & Safety Disclosures ────────────────────────────────────
st.markdown("---")
st.warning(
    "⚠️ MEDICAL DISCLAIMER & PROTOCOLS\n\n"
    "1. All generated outputs, report summaries, and profiles are intended strictly for local educational guidance and general references.\n"
    "2. If addressing any visible condition parameters or cutaneous symptoms (e.g., specific target rashes), ensure you log full textual characteristics description sets.\n"
    "3. Clinical responses employ careful evaluation frameworks (such as 'appears consistent with' or 'resembles') and outline at least 3 separate diagnostic possibilities or 3 medication alternatives to maintain cross-examination safety.\n"
    "4. Always cross-examine and verify the physical pharmaceutical labels, packaging, and factory instructions closely before applying or utilizing any compound dosages."
)
st.caption(
    "This operational system serves general informational training purposes and does not substitute "
    "for a physical diagnosis by a licensed clinical specialist."
)
