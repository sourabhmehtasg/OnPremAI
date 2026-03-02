# OnPremAI

Love this question. You’re thinking about the *right* layer now — not just “make it work”, but **owning the stack**.

I’ll do this in **three clean phases**, exactly as you asked:

1️⃣ **Fresh local Python environment setup (from zero)**
2️⃣ **Clarify LLaMA / GGUF / Ollama confusion (very important)**
3️⃣ **Fully local version of your notebook (code + explanations)**

No shortcuts, no magic steps.

---

# PHASE 1️⃣ — Clean local setup (from absolute scratch)

I’ll assume:

* **OS:** Windows / macOS / Linux (steps are identical except paths)
* **Python:** 3.10 or 3.11 (recommended)

---

## 🔹 Step 1: Create a new project folder

Pick a place you like (Desktop / Projects / etc).

```bash
mkdir ai_study_buddy_local
cd ai_study_buddy_local
```

This folder will contain:

```
ai_study_buddy_local/
│
├── venv/
├── models/
├── data/
├── output/
└── main.py
```

We’ll create these step by step.

---

## 🔹 Step 2: Create a Python virtual environment

### Why?

* Isolates dependencies
* Prevents conflicts
* Required for any serious ML work

### Command

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You should now see:

```text
(venv)
```

---

## 🔹 Step 3: Upgrade pip (important)

```bash
pip install --upgrade pip
```

---

## 🔹 Step 4: Install required libraries (with WHY)

### 📦 Core ML & LLM stack

```bash
pip install llama-cpp-python
```

**Why**

* Runs LLaMA / Mistral models **locally**
* Uses CPU or GPU
* Loads `.gguf` model files

---

### 📦 Embeddings (free & local)

```bash
pip install sentence-transformers
```

**Why**

* Converts text → vectors
* Replaces Gemini embeddings
* Fully offline

---

### 📦 Vector database

```bash
pip install faiss-cpu
```

**Why**

* Fast similarity search
* Same role as in your Gemini version

---

### 📦 PDF parsing

```bash
pip install pypdf
```

**Why**

* Extracts text from PDFs
* Lightweight & reliable

---

### 📦 PDF generation (exam paper)

```bash
pip install reportlab
```

**Why**

* Professional PDF formatting
* Headers, footers, LaTeX-like layout

---

### 📦 Text splitting (LangChain utility only)

```bash
pip install langchain-text-splitters
```

**Why**

* Chunking logic
* We do NOT use LangChain LLMs anymore

---

### ✅ Optional (but recommended)

```bash
pip install tqdm
```

**Why**

* Progress bars for long operations

---

## 🔹 Step 5: Create project folders

```bash
mkdir models data output
```

---

# PHASE 2️⃣ — GGUF, LLaMA, Ollama (this is the “nitty gritty”)

You asked **exactly the right question here** 👇

> *“I am guessing this is Ollama installed locally, correct me if i am wrong?”*

### ❌ Short answer: **No, this is NOT Ollama**

Let me explain clearly.

---

## 🧠 What is `.gguf`?

**GGUF** is a **binary model file format** used by:

* `llama.cpp`
* `llama-cpp-python`

It contains:

* Model weights
* Quantization info
* Tokenizer
* Everything needed to run an LLM

Think of it as:

> `.gguf` ≈ `.exe` for LLMs

---

## 🧠 What is `llama-cpp-python`?

It is:

* A Python wrapper around `llama.cpp`
* Loads `.gguf` files directly
* Runs fully offline

This line:

```python
llm = Llama(model_path="mistral-7b-instruct.Q4_K_M.gguf")
```

Means:

> “Load this LLM file from disk and run inference locally.”

No Ollama involved.

---

## 🧠 Then what is Ollama?

Ollama is:

* A **model manager + API server**
* Wraps `llama.cpp` internally
* Hides `.gguf` details

Comparison:

| Feature            | llama-cpp-python | Ollama       |
| ------------------ | ---------------- | ------------ |
| Needs `.gguf` file | ✅ Yes            | ❌ No         |
| Python native      | ✅ Yes            | ❌ (HTTP API) |
| Fine control       | ✅ High           | ❌ Limited    |
| Best for apps      | ✅                | ⚠️ Depends   |

👉 For **learning + control**, `llama-cpp-python` is better.

---

## 📥 Where do you get `.gguf` models?

### Best source: **Hugging Face**

Search for:

```
TheBloke Mistral GGUF
```

### Example model (recommended)

🔗 **Mistral 7B Instruct (quantized)**
File:

```
mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### Download steps

1. Go to Hugging Face
2. Open model page
3. Download `.gguf`
4. Place it here:

```
ai_study_buddy_local/
└── models/
    └── mistral-7b-instruct.Q4_K_M.gguf
```

---

## 🧮 Quantization explained (important)

| Variant | RAM needed | Quality |
| ------- | ---------- | ------- |
| Q4_K_M  | ~5–6 GB    | ⭐⭐⭐⭐    |
| Q5      | ~7 GB      | ⭐⭐⭐⭐⭐   |
| Q8      | ~10–12 GB  | ⭐⭐⭐⭐⭐⭐  |

👉 **Q4_K_M** is the best balance.

---

# PHASE 3️⃣ — Fully local version of your notebook (code)

Now we convert **your exact notebook logic**, but:

❌ No Gemini
❌ No API keys
❌ No cloud

---

## 📄 `main.py` (FULL LOCAL VERSION)

```python
import os
import re
from datetime import datetime

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from llama_cpp import Llama

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Drawing, Line

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "models/mistral-7b-instruct.Q4_K_M.gguf"
PDF_PATH = "data/input.pdf"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# LOAD LOCAL LLM
# ---------------------------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8
)

# ---------------------------
# LOAD PDF
# ---------------------------
reader = PdfReader(PDF_PATH)
text = "\n".join([p.extract_text() for p in reader.pages])

# ---------------------------
# CHUNK TEXT
# ---------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_text(text)

# ---------------------------
# EMBEDDINGS
# ---------------------------
embedder = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedder.encode(chunks, show_progress_bar=True)

# ---------------------------
# FAISS INDEX
# ---------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

# ---------------------------
# RETRIEVE CONTEXT
# ---------------------------
query = "Core concepts and equations"
q_emb = embedder.encode([query])
_, idx = index.search(q_emb, k=10)
context = "\n\n".join([chunks[i] for i in idx[0]])

# ---------------------------
# PROMPT
# ---------------------------
prompt = f"""
You are a university professor writing a formal exam paper.

Based on the following CONTEXT, generate 5 MCQs.

FORMAT:
Q#. Question
A) ...
B) ...
C) ...
D) ...
ANSWER_LINE: X ||| Explanation

CONTEXT:
{context}
"""

# ---------------------------
# LLM GENERATION
# ---------------------------
response = llm(
    f"<s>[INST]{prompt}[/INST]</s>",
    max_tokens=1200
)["choices"][0]["text"]

# ---------------------------
# PDF OUTPUT
# ---------------------------
pdf_path = os.path.join(OUTPUT_DIR, "Exam_Paper.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=A4)

styles = getSampleStyleSheet()
story = []

story.append(Paragraph("Examination Paper", styles["Title"]))
story.append(Spacer(1, 20))

for line in response.split("\n"):
    if line.strip():
        story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 8))

doc.build(story)

print(f"✅ Exam paper generated: {pdf_path}")
```

---

# 🧠 What you gained by going local

✅ Zero API cost
✅ Full control
✅ Works offline
✅ No rate limits
✅ Deployable anywhere

---

## Next possible upgrades

If you want, next we can:

* 🔁 Add **answer key page again**
* ⚡ Speed up with GPU
* 🌐 Wrap it with **Streamlit UI**
* 📦 Convert to **Docker**
* 🔍 Improve retrieval quality

Just say the word.
