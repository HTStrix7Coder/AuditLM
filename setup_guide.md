# AuditLM — Friend Setup Guide (LM Studio)

This is the only guide you need to run AuditLM on a Windows laptop. No Python, no Ollama, no vLLM.

You will load a local model file, paste invoices, and get a **Valid** or **Flagged** verdict with a `<think>` reasoning trace.

---

## 1. What you should have received

| File | What it is |
|------|------------|
| `auditlm-3b-q4_k_m.gguf` | The model (~1.8 GB). **Required.** |
| `setup_guide.md` | This file |
| `inference/lmstudio-system-prompt.txt` | System prompt (copy-paste) |
| `inference/lmstudio-invoice-prompt.txt` | First test invoice (copy-paste) |

If you only got the `.gguf`, that is enough — the prompts are also copied below.

Put the GGUF somewhere easy, for example:

```
C:\AuditLM\auditlm-3b-q4_k_m.gguf
```

---

## 2. Laptop requirements

| | Recommended | Minimum |
|--|-------------|---------|
| OS | Windows 10/11 | Windows 10 |
| GPU | NVIDIA, 4 GB+ VRAM | CPU-only (slow) |
| RAM | 16 GB | 8 GB |
| Disk | ~3 GB free | ~3 GB free |

Install the latest [NVIDIA Game Ready / Studio driver](https://www.nvidia.com/Download/index.aspx) if you have an NVIDIA GPU. LM Studio uses that driver automatically.

---

## 3. Install LM Studio

1. Open [https://lmstudio.ai](https://lmstudio.ai)
2. Download **LM Studio for Windows**
3. Run the installer
4. Open **LM Studio** (no account required for local use)

---

## 4. Import the AuditLM model

1. In the left sidebar, click the **folder / My Models** icon
2. Click **Import Model** (or drag the `.gguf` into the window)
3. Select `auditlm-3b-q4_k_m.gguf`
4. Wait until the model appears in the list

**Check the file size.** It should be about **1.8 GB**. If it is much smaller, the download was incomplete — copy the file again.

LM Studio should treat this as a **Qwen2.5** chat model. If you see a template / chat format dropdown, choose **Qwen** or **ChatML**.

---

## 5. Open a chat and load the model

1. Click the **Chat** icon (speech bubble) in the left sidebar
2. At the top, **Select a model to load** → choose `auditlm-3b-q4_k_m`
3. Wait until the status shows the model is loaded (not “loading”)

### GPU (if you have NVIDIA)

When LM Studio asks how many layers to offload, pick **as many as fit** (often “Max” or all layers). For a 4 GB card, Q4 3B should fit.

If it is very slow, you are probably on CPU — open **Settings** and enable GPU acceleration.

---

## 6. Set the system prompt (do this once)

Find the **System Prompt** box. In current LM Studio it is usually:

- the field above the chat, or
- **Chat settings / gear icon** → **System Prompt**

Paste **exactly** this (also in `inference/lmstudio-system-prompt.txt`):

```
You are Audit-LM, an AI specialized in financial compliance, invoice verification, and quantitative audit reasoning.
Think step-by-step inside <think>...</think> tags and provide your final verdict (Valid or Flagged) or calculated answer.
```

Leave the system prompt there for every invoice. Do **not** put the invoice inside the system prompt.

---

## 7. Set inference parameters

Open **Inference parameters** (gear / sliders next to the chat). Use:

| Setting | Value |
|---------|-------|
| Context Length | **4096** |
| Temperature | **0.1** |
| Top P | **0.9** (default is fine) |
| Max Tokens / Max Response | **512** or **768** |

If you see “Repeat penalty”, leave it at default.

---

## 8. Run the first test invoice

In the **user** message box (the main chat input), paste **exactly** this (also in `inference/lmstudio-invoice-prompt.txt`):

```
You are Audit-LM, an AI specialized in financial compliance and invoice auditing.
Perform a full audit on the following invoice for mathematical accuracy, line-item consistency, and fraud detection.
Think step-by-step inside <think>...</think> tags and provide your final verdict (Valid or Flagged) along with the reason.

INVOICE
Invoice Number: INV-29042140
Date: 2026-08-05
Vendor: Sandoval Inc
Bill To: Enterprise Client Corp.

Line Items:
- Ameliorated bandwidth-monitored knowledgebase: 8 units @ $82.27 each = $630.26
- Integrated multimedia database: 8 units @ $154.42 each = $1235.36

Financial Summary:
Subtotal: $1893.52
Tax (5%): $94.68
Total: $1988.20
```

Press **Send**.

### What a good answer looks like

1. A block starting with `<think>` that checks line items, subtotal, tax, and total
2. After `</think>`, something like:

```
Verdict: Flagged
Reason: ...
```

This invoice is **fraudulent** (line-item math is wrong: 8 × $82.27 = $658.16, not $630.26). The model should **Flag** it.

If you get no `<think>` tags, go back to step 6 and confirm the system prompt is set, and that you pasted the **full user prefix** (the “You are Audit-LM…” paragraph), not only the invoice.

---

## 9. Audit your own invoices

Keep the **system prompt** from step 6.

For every new invoice, the **user** message must start with this prefix, then a blank line, then the invoice:

```
You are Audit-LM, an AI specialized in financial compliance and invoice auditing.
Perform a full audit on the following invoice for mathematical accuracy, line-item consistency, and fraud detection.
Think step-by-step inside <think>...</think> tags and provide your final verdict (Valid or Flagged) along with the reason.

INVOICE
...paste the invoice here...
```

Start a **new chat** (or clear history) if the previous invoice is confusing the model.

---

## 10. Second test (optional — should be Valid)

User message:

```
You are Audit-LM, an AI specialized in financial compliance and invoice auditing.
Perform a full audit on the following invoice for mathematical accuracy, line-item consistency, and fraud detection.
Think step-by-step inside <think>...</think> tags and provide your final verdict (Valid or Flagged) along with the reason.

INVOICE
Invoice Number: INV-80874563
Date: 2026-03-05
Vendor: Howe, Lucero and Snyder
Bill To: Enterprise Client Corp.

Line Items:
- Self-enabling scalable encoding: 6 units @ $223.62 each = $1341.72
- Organic composite support: 2 units @ $172.18 each = $344.36

Financial Summary:
Subtotal: $1686.08
Tax (8%): $134.89
Total: $1820.97
```

Expected: **Verdict: Valid** (math checks out).

---

## Troubleshooting

| What you see | What to do |
|--------------|------------|
| Model missing / import fails | Confirm the `.gguf` is ~1.8 GB and not still copying |
| “Out of memory” | Lower GPU layers, or close other apps; CPU-only also works but is slower |
| Very slow replies | Enable GPU offload; update NVIDIA driver |
| No `<think>` / random Qwen-style chat | Set the system prompt; user message must include the invoice prefix |
| Always “Valid” or always “Flagged” | Temperature 0.1; start a new chat; paste the full user prefix |
| Cuts off mid-sentence | Raise Max Tokens to 768 |
| Garbled special tokens | Chat template = **Qwen** / **ChatML**, not Llama |

---

## You do not need

- Python
- CUDA toolkit
- vLLM
- Ollama
- The training datasets or Unsloth folder

Those are only for training. Inference is: **LM Studio + this GGUF + the two prompts above.**
