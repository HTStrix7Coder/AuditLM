# AuditLM — Verifiable Financial Invoice Auditing

A lightweight LLM fine-tuned for **invoice fraud detection**, **financial QA**, and **step-by-step audit reasoning**. Built with Unsloth + GRPO on Qwen2.5 (1.5B / 3B), exported to GGUF for easy local inference via **LM Studio**, **Ollama**, or **vLLM**.

| Model | Accuracy | F1 | MCC |
|-------|----------|----|-----|
| GRPO 3B (recommended) | **52.5%** | 0.67 | 0.44 |
| GRPO 1.5B | 40.0% | 0.43 | 0.06 |
| SFT 1.5B | 32.5% | 0.58 | 0.21 |
| Baseline Qwen2.5-1.5B | 42.5% | 0.54 | 0.19 |

Eval set: 40 held-out samples (25 invoice audits, 8 FinQA, 7 financial reasoning). See [`eval_results/comprehensive_eval_report.md`](eval_results/comprehensive_eval_report.md).

---

## Quick Start — Inference Only (Ollama, Windows / Linux / macOS)

Best path for sharing with someone who only needs to **run** the model.

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/AuditLM.git
cd AuditLM
```

### 2. Get the GGUF weights (~1.8 GB)

Download `auditlm-3b-q4_k_m.gguf` and place it here:

```
models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf
```

See [`models/README.md`](models/README.md) for Git LFS, Hugging Face, or manual transfer options.

### 3. Install [Ollama](https://ollama.com/download)

### 4. Register the model

**Windows (PowerShell):**

```powershell
.\inference\setup_ollama.ps1
```

**Linux / macOS / WSL:**

```bash
chmod +x inference/setup_ollama.sh
./inference/setup_ollama.sh
```

Or manually:

```bash
ollama create auditlm -f inference/Modelfile
```

### 5. Run

```bash
# Interactive chat
ollama run auditlm

# Pretty demo client (stdlib only — no pip install needed)
python scripts/live_demo_vllm.py --url http://localhost:11434/v1 --model auditlm --example
python scripts/live_demo_vllm.py --url http://localhost:11434/v1 --model auditlm
```

Paste an invoice when prompted. The model returns a `<think>` reasoning trace and a **Valid** or **Flagged** verdict.

---

## Quick Start — LM Studio (GUI, easiest on Windows)

No terminal needed — good for demos and non-technical users.

1. Install [LM Studio](https://lmstudio.ai)
2. **Import** `models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf`
3. Open **Chat** — set **System** from [`inference/lmstudio-system-prompt.txt`](inference/lmstudio-system-prompt.txt)
4. Paste **User** message from [`inference/lmstudio-invoice-prompt.txt`](inference/lmstudio-invoice-prompt.txt) (or prefix + your invoice — see [`inference/PROMPTS.md`](inference/PROMPTS.md))
5. Set **context 4096**, **temperature 0.1**

Full walkthrough: [`inference/LM_STUDIO.md`](inference/LM_STUDIO.md)

**Optional API** (use with `live_demo_vllm.py`): start LM Studio's local server → `http://localhost:1234/v1`

```bash
python scripts/live_demo_vllm.py --url http://localhost:1234/v1 --model auditlm-3b-q4_k_m --example
```

---

## Inference — vLLM (Linux / WSL2)

vLLM does **not** support native Windows. Use WSL2 or Linux.

```bash
pip install "vllm>=0.20"

vllm serve models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf --port 8000

python scripts/live_demo_vllm.py --url http://localhost:8000/v1 --example
```

---

## Inference — Unsloth (GPU, LoRA adapter)

Loads the GRPO LoRA adapter in 4-bit (~3–4 GB VRAM). Requires NVIDIA GPU.

```bash
pip install -r requirements-train.txt
python scripts/live_demo.py              # GRPO 3B (default)
python scripts/live_demo.py --example    # 3 built-in showcase invoices
python scripts/live_demo.py --model 1.5b
```

---

## Project Structure

```
AuditLM/
├── README.md
├── requirements-train.txt       # Unsloth + GRPO training
├── requirements-inference.txt   # vLLM (optional)
├── .env.example                 # INFERENCE_URL, INFERENCE_MODEL
├── inference/
│   ├── PROMPTS.md               # Canonical prompt reference
│   ├── LM_STUDIO.md             # LM Studio GUI setup guide
│   ├── lmstudio-system-prompt.txt
│   ├── lmstudio-invoice-prompt.txt
│   ├── Modelfile                # Ollama model definition
│   ├── setup_ollama.sh          # Linux / WSL helper
│   └── setup_ollama.ps1         # Windows helper
├── models/
│   ├── README.md                # How to download GGUF / adapters
│   ├── auditlm_3b_gguf/         # GGUF for Ollama / vLLM (not in git)
│   └── auditlm_grpo_3b_final/   # LoRA adapter (not in git)
├── scripts/
│   ├── prompts.py               # Canonical SYSTEM + invoice user prompts
│   ├── _paths.py                # Repo-relative paths
│   ├── live_demo_vllm.py        # Demo client → Ollama / vLLM / LM Studio API
│   ├── live_demo.py             # Demo client → Unsloth local GPU
│   ├── train_sft.py             # SFT warmup
│   ├── train_grpo.py            # GRPO fine-tuning
│   ├── export_3b_gguf.py        # LoRA → GGUF export
│   └── comprehensive_eval.py    # Full eval suite
├── datasets/                    # Training data (large sets gitignored)
├── eval_results/                # Benchmark reports
└── docs/
    └── Audit-LM_Project_Plan.md
```

---

## Training (full pipeline)

Requires NVIDIA GPU, CUDA, and Unsloth.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt

# Build dataset
python scripts/build_unified_dataset.py

# SFT warmup → GRPO
python scripts/train_sft.py
python scripts/train_grpo.py

# Export GGUF for sharing
python scripts/export_3b_gguf.py

# Evaluate
python scripts/comprehensive_eval.py
```

---

## Push to GitHub

From the `AuditLM/` folder (or repo root if AuditLM is the whole repo):

```bash
cd AuditLM
git init
git add .
git commit -m "Initial AuditLM release: scripts, docs, Ollama inference setup"

# Optional: track GGUF with Git LFS (1.8 GB)
git lfs install
git lfs track "*.gguf"
git add .gitattributes models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf
git commit -m "Add GGUF model weights via LFS"

git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AuditLM.git
git push -u origin main
```

**Without LFS:** keep models gitignored and share the GGUF via Google Drive / Hugging Face. Point people to [`models/README.md`](models/README.md).

### Hugging Face (recommended for model hosting)

```bash
pip install huggingface_hub
huggingface-cli upload YOUR_ORG/auditlm-3b \
  models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf
```

---

## Environment Variables

Copy `.env.example` → `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_URL` | `http://localhost:11434/v1` | Ollama or vLLM OpenAI-compatible API |
| `INFERENCE_MODEL` | `auditlm` | Model name on the server |

Used by `live_demo_vllm.py` when `--url` / `--model` are omitted.

---

## Hardware

| Setup | VRAM / RAM | Notes |
|-------|------------|-------|
| LM Studio + Q4 GGUF | ~4 GB GPU or 8 GB RAM | Easiest GUI on Windows |
| Ollama + Q4 GGUF | ~4 GB GPU or 8 GB RAM | Best for scripts / API |
| vLLM + GGUF | ~4 GB GPU | Linux / WSL2 |
| Unsloth 4-bit | ~3–4 GB GPU | Loads LoRA + base model |

---

## License

Add your license here before public release.

## Citation

If you use this work, please cite the base model [Qwen2.5](https://huggingface.co/Qwen) and note GRPO fine-tuning with Unsloth.
