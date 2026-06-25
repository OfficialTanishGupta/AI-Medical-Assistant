# 🩺 AI Medical Assistant (Offline LLM + RAG + Voice Assistant)

An AI-powered Medical Assistant that runs **completely offline** using a local Large Language Model (LLM), Retrieval-Augmented Generation (RAG), a Medical Knowledge Base, and Voice Interaction capabilities.

This project enables users to ask medical questions, search medicine information, retrieve answers from trusted medical documents, and interact with the assistant using voice commands without requiring internet connectivity.

---

# 🚀 Features

## Implemented Features (Phase 1 - Phase 9)

### 🤖 Local AI Assistant

- Local Llama 3 Integration using Ollama
- Fully Offline Operation
- Privacy-Focused Design
- No External API Required

### 📚 Medical Knowledge Base

- Medical PDF Storage
- PDF Text Extraction
- Document Chunking
- Knowledge Retrieval System

### 🔍 Retrieval-Augmented Generation (RAG)

- Context-Aware Responses
- Source-Based Answers
- Reduced Hallucinations
- Domain-Specific Knowledge Retrieval

### 🗄️ Vector Database

- ChromaDB Integration
- Semantic Search
- Fast Information Retrieval
- Persistent Knowledge Storage

### 💊 Medicine Search

- Medicine Information Lookup
- Uses and Applications
- Side Effects Information
- Dosage References
- Medical Warnings

### 🖥️ Interactive Interface

- Streamlit Chat Interface
- Chat History
- User-Friendly Dashboard
- Medical Query Input

### 🎙️ Voice Assistant

- Speech-to-Text
- Voice Question Input
- Text-to-Speech Responses
- Hands-Free Interaction

---

# 🎯 Project Objective

Healthcare information is often inaccessible without internet connectivity or expensive subscription services.

This project aims to create a:

- Private
- Offline
- Intelligent
- Domain-Specific
- AI-Powered

Medical Assistant capable of providing reliable information from trusted medical resources stored locally.

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
                    Local Llama 3
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
      User Query                     Voice Query
```

---

# 🛠️ Technology Stack

## Artificial Intelligence

- Llama 3
- Ollama
- LangChain
- Sentence Transformers

## Vector Database

- ChromaDB

## Frontend

- Streamlit

## Backend

- Python

## Document Processing

- PyPDF

## Voice Processing

- SpeechRecognition
- Pyttsx3

---

# 📂 Project Structure

```text
AI-Medical-Assistant/
│
├── data/
│   └── medical_pdfs/
│
├── vector_db/
│
├── models/
│
├── src/
│   ├── app.py
│   ├── ingest.py
│   ├── rag.py
│   ├── voice_assistant.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/OfficialTanishGupta/AI-Medical-Assistant.git

cd AI-Medical-Assistant
```

---

## Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

---

## Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install chromadb
pip install pypdf
pip install sentence-transformers
pip install ollama
pip install SpeechRecognition
pip install pyttsx3
```

---

# 🦙 Local LLM Setup

This project uses a local Large Language Model through Ollama.

## Install Ollama

Visit:

https://ollama.com

Download and install Ollama.

---

## Pull Llama 3 Model

```bash
ollama pull llama3
```

---

## Verify Installation

```bash
ollama run llama3
```

Example Query:

```text
What is diabetes?
```

---

# 📚 Medical Knowledge Base Setup

Store your medical documents inside:

```text
data/medical_pdfs/
```

Recommended Sources:

- Pharmacology Books
- Medical Textbooks
- WHO Documentation
- Disease Reference Manuals
- Clinical Guidelines
- Medical Handbooks

---

# 📄 PDF Processing Pipeline

The ingestion system performs:

### 1. Load PDFs

Medical documents are loaded automatically.

### 2. Extract Text

All readable content is extracted.

### 3. Chunk Text

Large documents are divided into smaller sections.

### 4. Generate Embeddings

Embeddings are created for semantic understanding.

### 5. Store Embeddings

Embeddings are stored in ChromaDB.

---

# 🔎 Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

Benefits:

- Lightweight
- Fast
- Accurate Semantic Search
- Efficient Local Execution

---

# 🗄️ Vector Database

ChromaDB is used to store document embeddings.

Responsibilities:

- Embedding Storage
- Similarity Search
- Retrieval Optimization
- Persistent Knowledge Storage

---

# 🤖 Retrieval-Augmented Generation (RAG)

Instead of relying only on the LLM's training data:

```text
User Question
      │
      ▼
Retrieve Relevant Chunks
      │
      ▼
Inject Context into Prompt
      │
      ▼
Llama 3 Generates Response
      │
      ▼
Answer with Sources
```

Benefits:

- Accurate Responses
- Medical Context Awareness
- Reduced Hallucinations
- Updatable Knowledge Base

---

# 💊 Medicine Search Module

Users can search medicines directly.

Example:

```text
Paracetamol
```

Expected Output:

```text
Medicine Name

Uses

Recommended Applications

Warnings

Possible Side Effects

Dosage Information

Drug Interactions
```

Data is retrieved from the medical knowledge base.

---

# 🖥️ Streamlit Interface

The application provides a chatbot-style interface.

Features:

- Ask Medical Questions
- Search Medicines
- View AI Responses
- Source Display
- Conversation History

Run:

```bash
streamlit run src/app.py
```

Application URL:

```text
http://localhost:8501
```

---

# 🎙️ Voice Assistant Module

Voice Assistant enables hands-free interaction.

### Features

- Speak Questions
- Convert Speech to Text
- Generate AI Responses
- Convert Responses to Speech
- Real-Time Interaction

---

## Required Libraries

```bash
pip install SpeechRecognition
pip install pyttsx3
```

---

## Voice Workflow

```text
User Speaks
      │
      ▼
Speech Recognition
      │
      ▼
Text Conversion
      │
      ▼
RAG Pipeline
      │
      ▼
Llama 3 Response
      │
      ▼
Text-to-Speech Output
```

---

# ▶️ Running the Application

## Launch Streamlit UI

```bash
streamlit run src/app.py
```

---

## Launch Voice Assistant

```bash
python src/voice_assistant.py
```

---

# 📊 Development Roadmap

## ✅ Phase 1

Project Setup

## ✅ Phase 2

Local LLM Integration

## ✅ Phase 3

Medical Knowledge Base Setup

## ✅ Phase 4

PDF Extraction Pipeline

## ✅ Phase 5

Vector Database Creation

## ✅ Phase 6

RAG Pipeline

## ✅ Phase 7

Streamlit Chat Interface

## ✅ Phase 8

Medicine Search Feature

## ✅ Phase 9

Voice Assistant Integration

---

# 🔮 Upcoming Features

## 🚀 Phase 10

Medical Report Summarizer

- Blood Test Reports
- Prescriptions
- Medical Records
- Lab Reports

---

## 🚀 Phase 11

Medical LLM Fine-Tuning

Using:

- LoRA
- QLoRA
- Unsloth

Create:

```text
Medical-Llama-3
```

Domain-specialized Medical Model

---

## 🚀 Phase 12

Docker Deployment

Features:

- One-Click Setup
- Portable Deployment
- Easy Installation

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
