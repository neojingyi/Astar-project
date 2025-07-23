# ESG Report Extractor with Groq – README

This repository contains the first iteration of an **ESG report extraction** pipeline, powered by Groq’s LLM and a FAISS-backed semantic search. It provides tools to:

- **Convert** PDFs to plain text  
- **Chunk** and **index** text by ESG taxonomy headings  
- **Retrieve** relevant passages for GRI disclosures  
- **Invoke** a Groq LLM to extract or paraphrase answers  

---

## Table of Contents

1. [Features](#features)  
1. [Prerequisites](#prerequisites)  
1. [Installation](#installation)  
1. [Directory Structure](#directory-structure)  
1. [Usage](#usage)  
   - [1. PDF → Text (`pdf_to_text.py`)](#1-pdf--text-pdf_to_textpy)  
   - [2. Create FAISS Index (`text_to_index.py`)](#2-create-faiss-index-text_to_indexpy)  
   - [3. Chunk by GRI Headings (`semantic_chunker.py`)](#3-chunk-by-gri-headings-semantic_chunkerpy)  
   - [4. Run Extraction Prompts (`running_prompts.py`)](#4-run-extraction-prompts-running_promptspy)  
   - [5. Accuracy Verification (optional) (`pdf_accuracy_checker.py`)](#5-accuracy-verification-optional-pdf_accuracy_checkerpy)  
1. [Configuration](#configuration)  
1. [Environment Variables](#environment-variables)  
1. [Extending & Troubleshooting](#extending--troubleshooting)  
1. [License](#license)  

---

## Features

- **Robust PDF text extraction** via PyMuPDF (`pdf_to_text.py`).  
- **Semantic chunking** by GRI section headings (`semantic_chunker.py`).  
- **FAISS** vector index creation for fast retrieval (`text_to_index.py`).  
- **Groq LLM**–powered RetrievalQA chain for precise answer extraction (`running_prompts.py`).  
- **Answer verification** against source PDF pages (`pdf_accuracy_checker.py`).  

---

## Prerequisites

- Python 3.8+  
- Groq API credentials  
- System libraries: `pandas`, `fitz` (PyMuPDF), `faiss-cpu`, `sentence-transformers`, `langchain`  

---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/esg-extractor-groq.git
   cd esg-extractor-groq
   ```
2. **Create & activate** a virtual environment  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Structure

```
.
├── main.py                   # Orchestrator script
├── pdf_to_text.py            # PDF → text converter
├── semantic_chunker.py       # Chunk text by GRI headings
├── text_to_index.py          # Build FAISS index
├── running_prompts.py        # Extraction pipeline
├── pdf_accuracy_checker.py   # Verify answers against PDF
├── requirements.txt
└── README.md
```

---

## Usage

### 1. PDF → Text (`pdf_to_text.py`)

```bash
python pdf_to_text.py <pdf_folder> <output_text_folder>
```

### 2. Create FAISS Index (`text_to_index.py`)

```bash
python text_to_index.py <text_folder> <index_path>
```

### 3. Chunk by GRI Headings (`semantic_chunker.py`)

Imported by `text_to_index.py`:
```python
from semantic_chunker import chunk_by_gri_headings
chunks = chunk_by_gri_headings(raw_text)
```

### 4. Run Extraction Prompts (`running_prompts.py`)

```bash
python running_prompts.py
```
Interactive: select PDF, loads disclosures, runs FAISS search + Groq LLM, saves `extracted_disclosure_answers_with_pages.xlsx`.

### 5. Accuracy Verification (`pdf_accuracy_checker.py`)

```bash
python pdf_accuracy_checker.py
```
Checks extracted answers against PDF pages.

---

## Configuration

Edit paths and model names at the top of each script or pass as arguments where supported.

---

## Environment Variables

```bash
export GROQ_API_KEY="sk-..."
```

---

## Extending & Troubleshooting

- **Long PDFs**: adjust chunk size in `running_prompts.py`.  
- **Rate limits**: retry logic on 429 errors included.  
- **Missing splits**: tweak regex in `semantic_chunker.py`.  

---

## License

MIT License. See [LICENSE](LICENSE).
