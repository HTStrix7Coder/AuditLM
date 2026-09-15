#!/usr/bin/env python3
"""
AuditLM — Live Reasoning Demo (Interactive & Showcase)
======================================================
Loads the fine-tuned model with Unsloth 4-bit inference so it fits
comfortably in your RTX 4060 Ti (~3-4 GB VRAM).

Features:
  - Live streaming output (watch <think> reasoning unfold token-by-token)
  - Pre-packaged showcase examples (--example)
  - Interactive REPL to paste and audit custom invoices
  - Colored ANSI display for reasoning trace vs conclusion

Usage:
  python scripts/live_demo.py              # GRPO-3B (default, best)
  python scripts/live_demo.py --model 1.5b # GRPO-1.5B
  python scripts/live_demo.py --model sft  # SFT warmup
  python scripts/live_demo.py --example    # Run showcase examples
"""
import unsloth
import os
import re
import sys
import time
import argparse
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

from _paths import model_path
from prompts import SYSTEM_PROMPT, format_invoice_user_prompt

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS = {
    "3b": {
        "path": str(model_path("auditlm_grpo_3b_final")),
        "label": "Audit-LM GRPO-3B (Best — 52.5% Acc, MCC=0.44)",
    },
    "1.5b": {
        "path": str(model_path("auditlm_grpo_1.5b_final")),
        "label": "Audit-LM GRPO-1.5B (40.0% Acc, MCC=0.06)",
    },
    "sft": {
        "path": str(model_path("auditlm_sft_1.5b_final")),
        "label": "Audit-LM SFT-1.5B (32.5% Acc, MCC=0.21)",
    },
}

EXAMPLES = [
    {
        "name": "FRAUDULENT — Wrong line-item calculation",
        "expected": "Flagged",
        "invoice": """INVOICE
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
Total: $1988.20""",
    },
    {
        "name": "VALID — All math verified",
        "expected": "Valid",
        "invoice": """INVOICE
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
Total: $1820.97""",
    },
    {
        "name": "FRAUDULENT — Duplicate line item injection",
        "expected": "Flagged",
        "invoice": """INVOICE
Invoice Number: INV-46176792
Date: 2026-05-14
Vendor: Ramirez Ltd
Bill To: Enterprise Client Corp.

Line Items:
- Profit-focused zero-defect protocol: 3 units @ $279.40 each = $838.20
- Progressive empowering capacity: 4 units @ $177.22 each = $708.88
- Total fresh-thinking parallelism: 5 units @ $272.87 each = $1364.35
- Reduced grid-enabled paradigm: 1 units @ $64.69 each = $64.69
- Profit-focused zero-defect protocol: 3 units @ $279.40 each = $838.20

Financial Summary:
Subtotal: $2976.12
Tax (10%): $297.61
Total: $3273.73""",
    },
]

# ── ANSI Styling ───────────────────────────────────────────────────────────────
RESET = "\033[0m"
def _c(code, t): return f"\033[{code}m{t}{RESET}"
def green(t):   return _c("92", t)
def red(t):     return _c("91", t)
def cyan(t):    return _c("96", t)
def yellow(t):  return _c("93", t)
def bold(t):    return _c("1",  t)
def dim(t):     return _c("2",  t)

# ── Reasoning Streamer ─────────────────────────────────────────────────────────
class ReasoningStreamer(TextStreamer):
    """
    Colorizes tokens live: cyan for <think>...</think> reasoning trace,
    bold default text for final conclusion/verdict.
    """
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.inside_think = False
        self.accumulated = []

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.accumulated.append(text)
        if "<think>" in text:
            self.inside_think = True
        
        if self.inside_think:
            sys.stdout.write(cyan(text))
        else:
            sys.stdout.write(text)
        sys.stdout.flush()

        if "</think>" in text:
            self.inside_think = False

    def get_full_text(self):
        return "".join(self.accumulated)


def extract_verdict(text):
    text_lower = text.lower()
    match = re.search(r"verdict:\s*(valid|flagged)", text_lower)
    if match:
        return match.group(1).capitalize()
    content_after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).lower()
    if "verdict" in content_after_think:
        if "flagged" in content_after_think:
            return "Flagged"
        if "valid" in content_after_think:
            return "Valid"
    if "flagged" in text_lower:
        return "Flagged"
    if "valid" in text_lower:
        return "Valid"
    return "Unknown"


def build_prompt(tokenizer, invoice_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_invoice_user_prompt(invoice_text)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def audit_invoice(model, tokenizer, invoice_text, max_new_tokens=512):
    prompt = build_prompt(tokenizer, invoice_text)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    streamer = ReasoningStreamer(tokenizer, skip_prompt=True)

    print()
    print(bold("─" * 65))
    print(cyan(bold("  🧠  LIVE REASONING TRACE (<think>...</think>) & VERDICT:")))
    print(bold("─" * 65))

    t0 = time.time()
    with torch.inference_mode():
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    full_output = streamer.get_full_text()
    verdict = extract_verdict(full_output)

    print("\n" + bold("─" * 65))
    print(dim(f"  ⏱  Audit finished in {elapsed:.2f}s"))

    if verdict == "Flagged":
        print(f"  Parsed Verdict: {bold(red('🚩 Flagged'))}")
    elif verdict == "Valid":
        print(f"  Parsed Verdict: {bold(green('✅ Valid'))}")
    else:
        print(f"  Parsed Verdict: {bold(yellow('❓ Unknown'))}")

    return verdict, full_output


def run_showcase(model, tokenizer, max_new_tokens):
    print(bold("\n" + "=" * 65))
    print(bold("       🧾  AuditLM Live Showcase — 3 Built-in Invoices"))
    print(bold("=" * 65))

    results = []
    for idx, item in enumerate(EXAMPLES, 1):
        item_name = item["name"]
        header_str = f"[{idx}/{len(EXAMPLES)}] {item_name}"
        print(f"\n\n{bold(header_str)}")
        print(dim("Invoice Document:"))
        for line in item["invoice"].splitlines():
            print(dim(f"  {line}"))

        verdict, _ = audit_invoice(model, tokenizer, item["invoice"], max_new_tokens)
        correct = verdict.lower() == item["expected"].lower()
        results.append((item["name"], item["expected"], verdict, correct))

    print("\n\n" + bold("=" * 65))
    print(bold("               SHOWCASE SUMMARY"))
    print(bold("=" * 65))
    correct_total = sum(1 for *_, c in results if c)
    for name, exp, got, ok in results:
        status = green("✓ MATCH") if ok else red("✗ MISMATCH")
        print(f"  {status} | Expected: {exp:<7} | Got: {got:<7} | {name}")
    print(f"\n  Total Score: {correct_total}/{len(EXAMPLES)} correct\n" + "=" * 65 + "\n")


def run_interactive(model, tokenizer, max_new_tokens):
    print(bold("\n" + "=" * 65))
    print(bold("       🔍  AuditLM Interactive Auditor (REPL)"))
    print(bold("=" * 65))
    print(dim("  Paste an invoice below. End input with an empty line (Enter twice)."))
    print(dim("  Type 'quit' or 'exit' to stop.\n"))

    while True:
        try:
            print(cyan(bold("\n[Input Invoice] > ")))
            lines = []
            while True:
                line = input()
                if line.strip().lower() in {"quit", "exit", "q"}:
                    print(dim("\nExiting. Good audit!\n"))
                    return
                if line == "" and lines:
                    break
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            print(dim("\nExiting.\n"))
            return

        invoice_text = "\n".join(lines).strip()
        if not invoice_text:
            continue

        audit_invoice(model, tokenizer, invoice_text, max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="AuditLM Live Reasoning Demo")
    parser.add_argument(
        "--model",
        choices=["3b", "1.5b", "sft"],
        default="3b",
        help="Which model adapter to load (default: 3b)",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run 3 built-in showcase test cases",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=768,
        help="Maximum generation tokens (default: 768)",
    )
    args = parser.parse_args()

    model_info = MODELS[args.model]
    adapter_path = model_info["path"]

    if not os.path.exists(adapter_path):
        print(red(f"Error: Model path not found at {adapter_path}"))
        sys.exit(1)

    print()
    print(bold("=" * 65))
    print(bold("           🧾  AuditLM Live Reasoning Demo"))
    print(bold("=" * 65))
    print(f"  {bold('Model')}     : {model_info['label']}")
    print(f"  {bold('Adapter')}   : {adapter_path}")
    print(f"  {bold('VRAM Mode')} : 4-bit (Optimized for RTX 4060 Ti)")
    print()
    print(dim("Loading model & tokenizer into GPU memory..."))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=False,
    )
    FastLanguageModel.for_inference(model)
    print(green("✓ Model successfully loaded and ready for inference!\n"))

    if args.example:
        run_showcase(model, tokenizer, args.max_tokens)
    else:
        run_interactive(model, tokenizer, args.max_tokens)


if __name__ == "__main__":
    main()
