# 🩺 AI Medical Assistant (Offline LLM + RAG)

An AI-powered Medical Assistant that runs **completely offline** using a local Large Language Model (LLM), Retrieval-Augmented Generation (RAG), and a Medical Knowledge Base.

This project enables users to ask medical questions, search medicine information, and retrieve accurate answers from medical documents without requiring an internet connection.

---

# 🚀 Features

### Current Features (Phase 1 - 8)

✅ Local LLM Integration using Ollama

✅ Offline AI Chat Assistant

✅ Medical PDF Knowledge Base

✅ PDF Text Extraction

✅ Medical Document Chunking

✅ Vector Database Creation (ChromaDB)

✅ Retrieval-Augmented Generation (RAG)

✅ Streamlit Chat Interface

✅ Medicine Information Search

✅ Source-Based Responses

---

# 🎯 Project Goal

Most AI chatbots depend on cloud servers and internet connectivity.

This project aims to build a:

- Private
- Offline
- Fast
- Domain-Specific

Medical AI Assistant capable of answering questions using trusted medical resources stored locally.

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
Embeddings
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Retriever
      │
      ▼
Llama 3 (Local)
      │
      ▼
Generated Answer
      │
      ▼
Streamlit Interface
```

---

# 🛠️ Tech Stack

## AI & Machine Learning

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
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/AI-Medical-Assistant.git

cd AI-Medical-Assistant
```

---

## 2️⃣ Create Virtual Environment

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

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or

```bash
pip install streamlit
pip install langchain
pip install langchain-community
pip install chromadb
pip install pypdf
pip install sentence-transformers
pip install ollama
```

---

# 🦙 Local LLM Setup

This project uses a local Llama model through Ollama.

## Install Ollama

Visit:

https://ollama.com

Download and install Ollama for your operating system.

---

## Pull Llama Model

```bash
ollama pull llama3
```

---

## Test Model

```bash
ollama run llama3
```

Example:

```text
What is diabetes?
```

---

# 📚 Medical Knowledge Base

The assistant uses medical PDFs as its source of information.

Store PDFs inside:

```text
data/medical_pdfs/
```

Examples:

- Disease Reference Books
- Medical Handbooks
- Pharmacology Guides
- WHO Publications
- Medicine Reference Manuals

---

# 📄 PDF Ingestion Pipeline

The ingestion system performs:

### Step 1

Load PDFs

### Step 2

Extract text

### Step 3

Split into chunks

### Step 4

Prepare chunks for embedding generation

---

# 🔎 Embedding Generation

The project converts medical text into embeddings using:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

Benefits:

- Fast retrieval
- Semantic search
- Better medical context understanding

---

# 🗄️ Vector Database

Embeddings are stored in:

```text
ChromaDB
```

Purpose:

- Store vector embeddings
- Perform similarity search
- Retrieve relevant medical knowledge

---

# 🤖 Retrieval-Augmented Generation (RAG)

Instead of relying only on Llama's training data:

```text
User Question
      │
      ▼
Retriever
      │
      ▼
Medical Chunks
      │
      ▼
Llama 3
      │
      ▼
Answer
```

Benefits:

- More accurate answers
- Domain-specific knowledge
- Reduced hallucinations
- Easy knowledge updates

---

# 💊 Medicine Search Feature

Users can search medicine names.

Example:

```text
Paracetamol
```

Expected Output:

```text
Uses

Dosage Information

Warnings

Side Effects

Interactions
```

Information is retrieved directly from the medical knowledge base.

---

# 🖥️ Streamlit Interface

The application includes a chatbot-style UI.

Features:

- Ask Medical Questions
- Search Medicines
- Display AI Responses
- View Source Documents
- User-Friendly Interface

---

# ▶️ Run Application

```bash
streamlit run src/app.py
```

Application launches at:

```text
http://localhost:8501
```

---

# 📊 Development Progress

## Phase 1

- [x] Project Setup

## Phase 2

- [x] Local LLM Integration

## Phase 3

- [x] Medical Knowledge Base Setup

## Phase 4

- [x] PDF Extraction Pipeline

## Phase 5

- [x] Vector Database Creation

## Phase 6

- [x] RAG Pipeline

## Phase 7

- [x] Streamlit Interface

## Phase 8

- [x] Medicine Search Feature

---

# 🔮 Upcoming Features

### Phase 9

Voice Assistant

- Speech-to-Text
- Text-to-Speech

### Phase 10

Medical Report Summarizer

- Blood Reports
- Prescriptions
- Lab Reports

### Phase 11

Medical LLM Fine-Tuning

- LoRA
- QLoRA
- Unsloth

### Phase 12

Docker Deployment

- Portable Installation
- Production Ready

---

# ⚠️ Disclaimer

This project is intended for educational and research purposes only.

The AI Medical Assistant does not replace professional medical advice, diagnosis, or treatment.

Always consult a qualified healthcare professional before making medical decisions.

---

# 👨‍💻 Author

Tanish Gupta

AI | Machine Learning | LLMs | RAG | Android Development

GitHub:
https://github.com/OfficialTanishGupta

---

# ⭐ Support

If you find this project useful:

⭐ Star the repository

🍴 Fork the repository

🛠️ Contribute to improve the project