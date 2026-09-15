# AuditLM — LM Studio Setup (Windows / macOS / Linux)

LM Studio is the easiest **GUI** option: load the GGUF, paste invoices, no terminal required.

## What you need

| Item | Details |
|------|---------|
| GGUF file | `models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf` (~1.8 GB) |
| LM Studio | [https://lmstudio.ai](https://lmstudio.ai) |
| GPU (recommended) | NVIDIA with 4 GB+ VRAM |

---

## Step 1 — Install LM Studio

Download and install from [lmstudio.ai](https://lmstudio.ai). No account required for local use.

---

## Step 2 — Load the model

1. Open **LM Studio**
2. Go to the **Models** tab (folder icon on the left)
3. Click **Import model** (or drag-and-drop the `.gguf` file)
4. Select:
   ```
   auditlm-3b-q4_k_m.gguf
   ```
5. Wait for import/indexing to finish

LM Studio should auto-detect **Qwen2.5** chat format from the GGUF metadata.

---

## Step 3 — Configure chat settings

Open the **Chat** tab (speech bubble icon) and load the model if prompted.

### System prompt

Paste [`lmstudio-system-prompt.txt`](lmstudio-system-prompt.txt) into the **System** field.

### User message (invoice audits)

Paste [`lmstudio-invoice-prompt.txt`](lmstudio-invoice-prompt.txt) as the **User** message, or paste your own invoice with the user prefix from [`PROMPTS.md`](PROMPTS.md).

The model was trained with instructions in the **user** turn, not only in system. Do not move the invoice audit instructions entirely into the system prompt.

### Recommended parameters

| Setting | Value |
|---------|-------|
| Context length | **4096** |
| Temperature | **0.1** |
| Max tokens | **512–768** |

In LM Studio these are usually under **Inference parameters** or the gear icon in Chat.

---

## Step 4 — Run your first audit

Paste a test invoice as the **user** message. A ready-made example is in [`lmstudio-invoice-prompt.txt`](lmstudio-invoice-prompt.txt).

Expected output:

1. A `<think>...</think>` reasoning block
2. A final line like `Verdict: Flagged` or `Final Verdict: Valid`

---

## Optional — Use the AuditLM demo script with LM Studio

LM Studio can expose an **OpenAI-compatible API** so `live_demo_vllm.py` works too.

### 1. Start the local server

1. Open the **Developer** / **Local Server** tab in LM Studio
2. Load `auditlm-3b-q4_k_m.gguf` if not already loaded
3. Click **Start server**
4. Default URL: `http://localhost:1234/v1`

Note the **model identifier** shown in the server panel (often the GGUF filename).

### 2. Run the demo client

From the repo root:

```bash
python scripts/live_demo_vllm.py \
  --url http://localhost:1234/v1 \
  --model auditlm-3b-q4_k_m \
  --example
```

Replace `--model` with whatever name LM Studio shows in the server UI.

Or set environment variables (copy from `.env.example`):

```bash
export INFERENCE_URL=http://localhost:1234/v1
export INFERENCE_MODEL=auditlm-3b-q4_k_m
python scripts/live_demo_vllm.py --example
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Model won't load | Ensure GGUF is Q4_K_M (~1.8 GB), not a partial download |
| Garbled / no thinking tags | Set system prompt; confirm Qwen2.5 template is selected |
| Slow on CPU | Enable GPU offload in LM Studio (Settings → GPU) |
| API connection refused | Start Local Server in LM Studio before running the Python client |
| Wrong model name in API | Copy the exact ID from the LM Studio server panel |

---

## LM Studio vs Ollama

| | LM Studio | Ollama |
|--|-----------|--------|
| Interface | GUI | Terminal / API |
| Best for | Demos, non-technical users | Scripts, automation |
| Windows | Native | Native |
| Same GGUF | Yes | Yes |

Both use the same `auditlm-3b-q4_k_m.gguf` file — pick whichever your friend prefers.
