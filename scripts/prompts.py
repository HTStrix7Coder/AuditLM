"""
Canonical AuditLM prompts — single source of truth.

Training/eval layout (match this at inference):
  system → SYSTEM_PROMPT
  user   → INVOICE_USER_PROMPT_PREFIX + invoice body   (invoice audits)
  user   → task-specific prompt from dataset            (FinQA / reasoning)
"""

# Used as the chat "system" message in SFT, GRPO, eval, and inference servers.
SYSTEM_PROMPT = (
    "You are Audit-LM, an AI specialized in financial compliance, invoice verification, "
    "and quantitative audit reasoning.\n"
    "Think step-by-step inside <think>...</think> tags and provide "
    "your final verdict (Valid or Flagged) or calculated answer."
)

# Prepended to raw invoice text for invoice-audit tasks (matches auditlm_train.json).
INVOICE_USER_PROMPT_PREFIX = (
    "You are Audit-LM, an AI specialized in financial compliance and invoice auditing.\n"
    "Perform a full audit on the following invoice for mathematical accuracy, line-item "
    "consistency, and fraud detection.\n"
    "Think step-by-step inside <think>...</think> tags and provide "
    "your final verdict (Valid or Flagged) along with the reason.\n\n"
)


def format_invoice_user_prompt(invoice_text: str) -> str:
    """Build the user message for an invoice audit (training-aligned)."""
    body = invoice_text.strip()
    if body.startswith("You are Audit-LM"):
        return body
    return f"{INVOICE_USER_PROMPT_PREFIX}{body}\n"
