# 🩺 AI Medical Assistant (Offline LLM + RAG + Voice Assistant)

An AI-powered Medical Assistant that runs **completely offline** using a local Large Language Model (LLM), Retrieval-Augmented Generation (RAG), a Medical Knowledge Base, and Voice Interaction capabilities.

This project enables users to ask medical questions, search medicine information, extract data from laboratory reports, and interact with the assistant using hands-free voice commands without requiring internet connectivity or external API keys.

---

# 🚀 Features

## Implemented Features (Phase 1 - Phase 10)

### 🤖 Local AI Assistant

- **Local Llama 3 Integration** via Ollama.
- **100% Offline Operation** ensuring total medical data privacy.
- **Privacy-Focused Design** where no medical data ever leaves your computer hard drive.
- **No External API Keys** or cloud subscriptions required.

### 📚 Medical Knowledge Base & RAG

- **Persistent Vector Store** powered by ChromaDB.
- **Fast Similarity Search** using the `all-MiniLM-L6-v2` embedding model.
- **Context-Aware Responses** that ground Llama 3's outputs directly to your uploaded text files.
- **Reduced Hallucinations** by enforcing strict factual fallback rules.

### 💊 Medicine Search Module

- **Pharmacology Lookup** for generic or brand medication names.
- **Structured Outputs** dividing references cleanly into Uses, Side Effects & Warnings, and Dosage Information.

### 🎙️ Voice Assistant Module

- **Speech-to-Text Input** allows users to click a button and vocalize their symptoms.
- **Text-to-Speech Output** reads the final medical or prescription analysis aloud using clear offline audio synthesis.

### 📄 Medical Report Summarizer (New)

- **PDF Extraction Pipeline** parses underlying text layers out of uploaded blood panels, prescriptions, or laboratory results.
- **Structured Laboratory Breakdown** providing a document summary, tracking out-of-range biomarker values, providing clinical observations, and outputting safety-hedged next-step recommendations.

### 🖥️ Cyber-Blue Dashboard UI

- **Modular Layout Split** providing real-time system status indicators, overall query counters, and tab-free workspace navigation.

---

# 🏗️ System Architecture

```text
                     Medical PDFs
                          │
                          ▼
                  PDF Extraction
                          │
                          ▼
                    Text Chunking
                          │
                          ▼
                 Embedding Creation
                          │
                          ▼
                    ChromaDB Store
                          │
                          ▼
                      Retriever
                          │
                          ▼
                    Local Llama 3 ◄─── Uploaded Lab Reports (PDF)
                          │
                          ▼
                   Generated Answer
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
  Streamlit Interface             Voice Assistant
         │                                 │
         ▼                                 ▼
    User Query /                       Voice Output
    Lab PDF Upload                     (Reads Aloud)
```

---

# 🛠️ Technology Stack

- **Large Language Model Engine:** Llama 3 via Ollama
- **Embeddings Pipeline:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Storage Database:** ChromaDB
- **Frontend Workspace Dashboard:** Streamlit
- **Document Processing Tools:** PyPDF (`PdfReader`)
- **Voice / Audio Automation:** SpeechRecognition, Pyttsx3, PyAudio
- **Core Runtime Environment:** Python 3.10+ / 3.11+

---

# 📂 Project Structure

```text
AI-Medical-Assistant/
│
├── chroma_db/               # Persistent ChromaDB collection binaries
├── data/
│   └── medical_guide.pdf    # Core knowledge base documents
│
├── models/                  # Optional local model file storage
├── vector_db/
│   └── vector_db.py         # DB Ingestion, chunking, and verification tests
│
├── src/
│   ├── app.py               # Main Streamlit UI layout and app workspace
│   ├── ingest.py            # PDF text layer parsing and extraction functions
│   ├── rag.py               # Core RAG retrieval loop and prompt handlers
│   └── voice_assistant.py   # Offline audio recorder and speech-to-text drivers
│
├── requirements.txt         # Project runtime dependency list
├── README.md                # System engineering documentation
└── .gitignore               # Venv, cache, and DB file tracking rules
```

---

# ⚙️ Installation & Workspace Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com
cd AI-Medical-Assistant
```

### Step 2: Establish a Virtual Environment

```bash
python -m venv venv
```

- **Windows Activation:** `venv\Scripts\activate`
- **Linux / Mac Activation:** `source venv/bin/activate`

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

_Note: If installing `pyaudio` fails on Windows due to build tools, run `pip install pipwin` followed by `pipwin install pyaudio` to map audio compilation binaries automatically._

---

# 🦙 Local Model Dependencies Setup

### 1. Download Ollama Engine

Head to [Ollama's Official Portal](https://ollama.com) and install the local background runtime assistant for your matching OS architecture.

### 2. Pull down Llama 3 Weights

Open up an independent, non-virtualized system console window and run:

```bash
ollama pull llama3
```

### 3. Verify Local Execution

Ensure the model is awake and responding locally:

```bash
ollama run llama3
# Test Query: "Hello" -> Once verified, type "/exit" to leave.
```

---

# ▶️ Running the Application

### 1. Start Your Database Pipeline (If collection is uninitialized)

To populate your `chroma_db/` folder with vectors generated directly from your target reference text, execute the ingestion file from your project root:

```bash
python vector_db/vector_db.py
```

### 2. Launch the Streamlit Ecosystem

Run the layout orchestration code using the runtime server:

```bash
streamlit run src/app.py
```

Your operating system will open up an offline application dashboard container instance located automatically at: `http://localhost:8501`

---

# 📊 Development Roadmap

- [x] **Phase 1:** Core Repository Architecture Setup
- [x] **Phase 2:** Local Ollama Llama 3 Integration
- [x] **Phase 3:** Data Directory Structuring
- [x] **Phase 4:** PyPDF Layer Text Extraction Pipelines
- [x] **Phase 5:** ChromaDB Vector Storage Generation
- [x] **Phase 6:** Context-Aware RAG Fallback Core Engines
- [x] **Phase 7:** Streamlit Chat Interface Implementation
- [x] **Phase 8:** Structured Medicine Profile Lookups
- [x] **Phase 9:** SpeechRecognition and Pyttsx3 Modular Integration
- [x] **Phase 10:** Medical Report Summarizer & Biomarker Extraction

---

# 🔮 Upcoming Features & Next Steps

### 🚀 Phase 11 — Medical LLM Fine-Tuning

Transition away from pure background RAG prompting. We will train a highly domain-specialized **Medical-Llama-3** model directly using parameter-efficient adapters:

- **Frameworks:** Unsloth / Hugging Face TRL / PEFT
- **Techniques:** 4-bit Quantization, LoRA, and QLoRA
- **Datasets:** Convert local medical guides into structured clinical instruction pairs (`dataset.json`).

### 🚀 Phase 12 — Docker Containers Deployment

Wrap up the local workspace inside a portable, repeatable ecosystem configuration layer:

- **Features:** One-click environment setups, volume mapping parameters for stored vector graphs, and network exposures to tie directly with external desktop systems.

---

# 📈 Future Enhancements

- Android Application
- Multi-Language Support
- Medical Image Analysis
- OCR for Prescriptions
- Disease Prediction Module
- Healthcare Recommendation Engine
- Hospital Finder Integration
- Emergency Guidance Module

---

# ⚠️ Disclaimer

This project is intended solely for educational, research, and demonstration purposes.

The AI Medical Assistant does not replace professional medical advice, diagnosis, treatment, or healthcare services.

Always consult a qualified healthcare professional before making medical decisions.

---

# 👨‍💻 Author

**Tanish Gupta**

AI Engineer | Machine Learning Enthusiast | LLM Developer | RAG Systems Builder | Android Developer

GitHub:

https://github.com/OfficialTanishGupta

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the repository

🛠️ Contribute to improve the project

📢 Share with the AI and Open Source Community

---

# 🌟 Project Vision

To build a fully offline, privacy-focused, domain-specialized AI Medical Assistant that can provide reliable medical information anywhere, anytime, without dependence on cloud services or internet connectivity.
