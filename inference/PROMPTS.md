# AuditLM Prompt Reference

All inference paths should match **training/eval** message layout.

## Invoice audits

| Role | Content | File |
|------|---------|------|
| **System** | `SYSTEM_PROMPT` | [`../scripts/prompts.py`](../scripts/prompts.py) · [`lmstudio-system-prompt.txt`](lmstudio-system-prompt.txt) · [`Modelfile`](Modelfile) |
| **User** | `INVOICE_USER_PROMPT_PREFIX` + invoice body | [`../scripts/prompts.py`](../scripts/prompts.py) · [`lmstudio-invoice-prompt.txt`](lmstudio-invoice-prompt.txt) (example) |

### System prompt (exact)

```
You are Audit-LM, an AI specialized in financial compliance, invoice verification, and quantitative audit reasoning.
Think step-by-step inside <think>...</think> tags and provide your final verdict (Valid or Flagged) or calculated answer.
```

### User prefix for invoices (exact — then paste invoice)

```
You are Audit-LM, an AI specialized in financial compliance and invoice auditing.
Perform a full audit on the following invoice for mathematical accuracy, line-item consistency, and fraud detection.
Think step-by-step inside <think>...</think> tags and provide your final verdict (Valid or Flagged) along with the reason.

```

## LM Studio

1. **System prompt** → paste `lmstudio-system-prompt.txt`
2. **User message** → paste `lmstudio-invoice-prompt.txt` (or prefix + your invoice)

Do **not** put the full audit checklist only in the system prompt — the model was trained with instructions in the **user** turn.

## Ollama

`Modelfile` sets `SYSTEM` to `SYSTEM_PROMPT`. Demo scripts add the user prefix automatically via `format_invoice_user_prompt()`.

## Other task types (FinQA / financial reasoning)

Task-specific instructions live in the dataset `prompt` field; system message is still `SYSTEM_PROMPT`. See `scripts/build_unified_dataset.py`.

## Baseline evaluation (intentionally different)

`scripts/evaluate_baseline.py` uses a **shorter** system prompt (no `<think>` instruction) to measure the raw base model — do not use that for production inference.
