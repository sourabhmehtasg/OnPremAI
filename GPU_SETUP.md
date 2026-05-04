🔙 [Back to Main Page](./README.md)

---

# 🚀 GPU Setup Playbook for Local LLMs (llama.cpp / llama-cpp-python)

> From “it runs” → to “it’s actually usable”

This guide documents a **real-world setup and optimization journey** for running LLMs locally with GPU acceleration.

It focuses on what actually breaks, what actually works, and what actually improves performance.

---

## 🎯 Goal

Run a quantized LLM locally with:

- ✅ GPU acceleration (CUDA)
- ✅ Usable latency (not just “it works”)
- ✅ Stable configuration for real applications

---

## 🧠 System Context

- GPU: RTX 2060 (6GB VRAM)
- Framework: `llama-cpp-python`
- Model: Mistral 7B (GGUF, quantized)
- OS: Windows 10/11

---

## ⚠️ Reality Check (Before You Start)

Running a local LLM:

| Stage | Difficulty |
|------|----------|
| CPU inference | Easy |
| GPU enablement | Medium |
| GPU optimization | Hard |

👉 Most setups stop at “GPU enabled”  
👉 Real value comes from “GPU optimized”

---

# 🧱 PHASE 1 — System Dependencies (CRITICAL)

This is where most setups fail.

---

## 1. Install NVIDIA Drivers

Ensure latest drivers are installed.

Verify:
```bash
nvidia-smi
````

---

## 2. Install CUDA Toolkit

Install a CUDA version compatible with:

* your GPU
* your build environment

👉 Recommended: CUDA 11.8 or stable 12.x version

---

## 🧩 3. CRITICAL — Use Correct Build Environment (Windows)

This is the most commonly missed step.

### ❗ Problem

Using normal CMD / PowerShell leads to:

* CUDA not detected during build
* CPU-only fallback (silent)
* build failures

---

## ✅ Solution: Use "x64 Native Tools Command Prompt for VS 2022"

Open:

> **x64 Native Tools Command Prompt for VS 2022**

Start Menu → search → open it

---

## 🧠 Why this matters

This terminal:

* Loads MSVC compiler (`cl.exe`)
* Configures:

  * CMake
  * Windows SDK
  * linker paths
* Aligns correctly with CUDA toolchain

👉 Without this, CUDA builds often fail or fallback silently

---

## ⚠️ CUDA 12.x Note

CUDA 12.x can be unstable with mismatched compilers.

This prompt ensures:

* correct architecture (x64)
* compatible compiler version
* proper linking

---

## 🚀 REQUIRED WORKFLOW (DO NOT SKIP)

### Step 1 — Open correct terminal

> x64 Native Tools Command Prompt for VS 2022

---

### Step 2 — Navigate to project

```bash
cd your_project_folder
```

---

### Step 3 — Create virtual environment

```bash
python -m venv venv
```

---

### Step 4 — Activate venv (same terminal)

```bash
venv\Scripts\activate
```

👉 Do NOT switch terminals after this

---

## 4. Install Microsoft C++ Build Tools

If not already installed:

Download:
[https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Select:

* Desktop development with C++

Ensure:

* MSVC v143 (or latest)
* Windows SDK
* CMake

Restart system after install.

---

## 🔍 Verify Toolchain

```bash
cl
cmake --version
```

---

# ⚙️ PHASE 2 — Install llama-cpp-python (GPU Enabled)

Inside the same VS2022 terminal + active venv:

```bash
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1

pip install llama-cpp-python --no-cache-dir
```

---

## 🔥 Common Failures

### ❌ CMake not found

👉 Install via MSVC tools

### ❌ CUDA not detected

👉 Check PATH and CUDA install

### ❌ CPU fallback

👉 Wrong terminal used

### ❌ Build errors

👉 Restart system, reinstall build tools

---

# 📦 PHASE 3 — Model Setup (GGUF)

## What is GGUF?

Binary format used by llama.cpp

👉 Think:

> `.gguf` = executable model

---

## Model Choice

Example:

* Mistral 7B Instruct
* Quantization: Q4_K_M

---

## Why Q4_K_M?

| Type   | VRAM    | Speed | Quality  |
| ------ | ------- | ----- | -------- |
| Q4_K_M | ~5–6 GB | Fast  | Balanced |

👉 Ideal for RTX 2060

---

# 🚀 PHASE 4 — CPU vs GPU Reality

## CPU

* ~5–10 tokens/sec
* usable for testing only

---

## GPU (after setup)

* ~37–40 tokens/sec
* ~7–8x improvement

👉 This is where system becomes usable

---

# ⚙️ PHASE 5 — What Actually Moves the Needle

GPU ≠ performance

---

## 1. n_gpu_layers (BIGGEST IMPACT)

```python
n_gpu_layers=26
```

---

## 2. n_batch

```python
n_batch=512
```

---

## 3. Threads

```python
n_threads=8
```

---

## 4. Context

```python
n_ctx=1024
```

---

# 🧪 PHASE 6 — Validation

## Check logs

* tokens/sec ~30+
* prompt eval fast
* GPU visible

---

## Example Metrics

| Metric       | Value            |
| ------------ | ---------------- |
| Tokens/sec   | ~37–40           |
| Prompt speed | ~700–900 tok/sec |

---

## CUDA Graphs Disabled?

Not an error.

Older GPUs don’t support it.

---

# 🧠 Key Learnings

* GPU enabled ≠ GPU optimized
* environment matters more than install
* tuning > setup
* CPU baseline is misleading

---

# 📈 Outcome

You now have:

* local inference engine
* GPU acceleration
* no API dependency
* production-ready base

---

# 🔜 Next Upcoming playbooks

* batch processing
* API layer
* Docker GPU
* smaller models

---
