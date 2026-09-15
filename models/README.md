# Model Weights

Model binaries are **not committed** to Git by default (too large). Pick one option below.

## Recommended for inference: GGUF (1.8 GB)

| File | Purpose |
|------|---------|
| `auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf` | Best model — GRPO 3B, Q4_K_M — use with **Ollama** or **vLLM** |

Place the file at:

```
models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf
```

### Get the GGUF

1. **From the project author** — USB / Google Drive / shared link
2. **Git LFS** (if enabled on the repo):
   ```bash
   git lfs install
   git lfs pull
   ```
3. **Hugging Face** (after upload):
   ```bash
   huggingface-cli download YOUR_ORG/auditlm-3b-q4_k_m auditlm-3b-q4_k_m.gguf \
     --local-dir models/auditlm_3b_gguf
   ```

## LoRA adapters (optional — Unsloth inference / re-training)

| Folder | Base model | Size |
|--------|------------|------|
| `auditlm_grpo_3b_final/` | Qwen2.5-3B-Instruct | ~130 MB |
| `auditlm_grpo_1.5b_final/` | Qwen2.5-1.5B-Instruct | ~122 MB |
| `auditlm_sft_1.5b_final/` | Qwen2.5-1.5B-Instruct | ~86 MB |

Used by `scripts/live_demo.py` and training scripts. Base weights are pulled from Hugging Face on first run.

## Export GGUF yourself

If you have the LoRA adapter and Unsloth installed:

```bash
python scripts/export_3b_gguf.py
```
