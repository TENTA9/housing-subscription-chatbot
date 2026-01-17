# 🏠 Korean Housing Subscription Q&A Chatbot (RAG)

A Domain-Specific RAG (Retrieval-Augmented Generation) system designed to answer complex inquiries regarding South Korean housing subscription laws and guidelines. This project overcomes the limitations of keyword-based search by implementing a hybrid retrieval system optimized for Korean legal texts.

## 📌 Project Overview
- **Project Type:** Graduate School Term Project
- **Role:** **AI Engineer** & Full-Stack Developer (Streamlit)
- **Tech Stack:** Python, LangChain, OpenAI, FAISS, Kiwi (Korean Morphological Analysis), Streamlit
- **Objective:** Solve information asymmetry in the housing market by providing accurate, evidence-based answers from vast legal documents.

## 📂 Project Documents
* 📄 **Project Report:** [View Final Report PDF](./docs/Term_Project_Report.pdf)

## 🛠 My Main Contributions

I designed the entire **RAG pipeline optimized for Korean legal documents**, focusing on retrieval accuracy and hallucination reduction.

### 1. Advanced Retrieval Engine (Hybrid Search)
* **Ensemble Retriever:** Combined **KiwiBM25** (Keyword-based) and **FAISS** (Semantic-based) to handle both exact legal terminology and contextual queries effectively.
* **Morphological Analysis:** Utilized `Kiwi` tokenizer (`kiwipiepy`) to accurately process Korean compound nouns in legal texts, significantly improving keyword matching performance.

### 2. Precision Optimization (Re-ranking)
* **BGE-M3 Reranker:** Integrated `BAAI/bge-m3` model to re-rank the retrieved documents. This step filters out irrelevant contexts that might share keywords but lack semantic relevance, ensuring high-quality context for the LLM.

### 3. Domain-Specific Data Pipeline
* **Preprocessing:** Built a custom PDF parsing pipeline (`pdfplumber`) to handle the complex layout of government handbooks.
* **Metadata filtering:** Structured chunks with metadata to allow source tracking and precise filtering.

### 4. Interactive Chat Interface
* **Streamlit UI:** Developed a user-friendly chatbot interface (`streamlit_app.py`) that not only answers questions but also provides **source citations** (document name, page content) to enhance trust.

## 📁 Repository Structure
```text
housing-subscription-chatbot
 ├── docs/                      # Final Project Report
 ├── pdfs/                      # Legal documents & Guidelines (Raw Data)
 ├── housing_rag.py             # Core RAG Logic (Indexing, Retrieval, Generation)
 ├── streamlit_app.py           # Chatbot Interface
 ├── requirements.txt
 └── README.md