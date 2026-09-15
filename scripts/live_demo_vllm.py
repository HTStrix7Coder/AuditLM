#!/usr/bin/env python3
"""
AuditLM — vLLM Live Reasoning Demo (Properly Formatted & Streaming)
==================================================================
Streams the audit reasoning token-by-token from vLLM:
- Visualizes <think>...</think> reasoning traces in a dedicated panel
- Highlights the final Verdict (Flagged / Valid) with color-coded badges
- Seamlessly connects to your running `vllm serve` server (default http://localhost:8000/v1)
  or can run embedded in-process.

Usage:
    # Connects to your running vLLM server:
    python scripts/live_demo_vllm.py --example     # Run 3 built-in showcase tests
    python scripts/live_demo_vllm.py               # Interactive REPL (paste invoices)
    python scripts/live_demo_vllm.py --invoice "..." # Audit one-liner invoice
"""

import os
import re
import sys
import time
import json
import argparse
import urllib.request
import urllib.error

from _paths import model_path
from prompts import SYSTEM_PROMPT, format_invoice_user_prompt

# ── ANSI Colour Helpers ────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
BG_RED  = "\033[41m\033[97m"
BG_GRN  = "\033[42m\033[97m"

def c(code, text): return f"{code}{text}{RESET}"
def bold(t):    return c(BOLD, t)
def dim(t):     return c(DIM, t)
def green(t):   return c(GREEN, t)
def red(t):     return c(RED, t)
def cyan(t):    return c(CYAN, t)
def yellow(t):  return c(YELLOW, t)
def magenta(t): return c(MAGENTA, t)

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_SERVER_URL = os.environ.get("INFERENCE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL_NAME = os.environ.get("INFERENCE_MODEL", "auditlm")
DEFAULT_GGUF_MODEL = str(model_path("auditlm_3b_gguf", "auditlm-3b-q4_k_m.gguf"))

EXAMPLES = [
    {
        "name":     "FRAUDULENT — Wrong line-item calculation ($27.90 stealth padding)",
        "expected": "Flagged",
        "invoice": """\
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
Total: $1988.20""",
    },
    {
        "name":     "VALID — All calculations mathematically correct",
        "expected": "Valid",
        "invoice": """\
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
Total: $1820.97""",
    },
    {
        "name":     "FRAUDULENT — Duplicate line item detected",
        "expected": "Flagged",
        "invoice": """\
INVOICE
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
    {
        "name":     "FRAUDULENT — Subtotal inflation (Stealth ghost surcharge)",
        "expected": "Flagged",
        "invoice": """\
INVOICE
Invoice Number: INV-91028341
Date: 2026-09-02
Vendor: Apex Cloud Solutions
Bill To: Enterprise Client Corp.

Line Items:
- Enterprise Server Hosting: 2 units @ $350.00 each = $700.00
- Dedicated Load Balancer: 1 units @ $150.00 each = $150.00

Financial Summary:
Subtotal: $1050.00
Tax (10%): $105.00
Total: $1155.00""",
    },
    {
        "name":     "FRAUDULENT — Tax rate manipulation (10% stated, but charged 20%)",
        "expected": "Flagged",
        "invoice": """\
INVOICE
Invoice Number: INV-77182904
Date: 2026-07-19
Vendor: Metro Logistics Group
Bill To: Enterprise Client Corp.

Line Items:
- Freight Haulage Fleet Dispatch: 4 units @ $300.00 each = $1200.00

Financial Summary:
Subtotal: $1200.00
Tax (10%): $240.00
Total: $1440.00""",
    },
    {
        "name":     "VALID — Cents rounding boundary edge case",
        "expected": "Valid",
        "invoice": """\
INVOICE
Invoice Number: INV-55419082
Date: 2026-04-11
Vendor: Precision Parts Co.
Bill To: Enterprise Client Corp.

Line Items:
- High-tolerance titanium flange: 3 units @ $33.33 each = $99.99
- Industrial synthetic gasket: 4 units @ $12.50 each = $50.00

Financial Summary:
Subtotal: $149.99
Tax (7.5%): $11.25
Total: $161.24""",
    },
    {
        "name":     "FRAUDULENT — Line-item multiplication error ($25.00 unit arithmetic padding)",
        "expected": "Flagged",
        "invoice": """\
INVOICE
Invoice Number: INV-10928345
Date: 2026-06-18
Vendor: Global Hardware Supply
Bill To: Enterprise Client Corp.

Line Items:
- Precision Power Supply Units: 5 units @ $45.00 each = $250.00
- Cat6 Shielded Ethernet Spool: 2 units @ $60.00 each = $120.00

Financial Summary:
Subtotal: $370.00
Tax (10%): $37.00
Total: $407.00""",
    },
    {
        "name":     "FRAUDULENT — Grand total summation mismatch ($100 stealth pad on total)",
        "expected": "Flagged",
        "invoice": """\
INVOICE
Invoice Number: INV-66291043
Date: 2026-08-22
Vendor: Zenith Office Outfitters
Bill To: Enterprise Client Corp.

Line Items:
- Ergonomic Task Chairs: 4 units @ $125.00 each = $500.00

Financial Summary:
Subtotal: $500.00
Tax (10%): $50.00
Total: $650.00""",
    },
    {
        "name":     "VALID — 4-item enterprise infrastructure invoice with odd decimals",
        "expected": "Valid",
        "invoice": """\
INVOICE
Invoice Number: INV-33918274
Date: 2026-05-30
Vendor: Northern Cloud & Fiber
Bill To: Enterprise Client Corp.

Line Items:
- Fiber Backbone Transit: 10 units @ $15.50 each = $155.00
- Managed Firewall Instance: 3 units @ $210.00 each = $630.00
- Cold Storage Retention: 12 units @ $8.75 each = $105.00
- DNS Anycast Routing: 1 units @ $45.00 each = $45.00

Financial Summary:
Subtotal: $935.00
Tax (5%): $46.75
Total: $981.75""",
    },
    {
        "name":     "FRAUDULENT — Phantom zero quantity line item billed",
        "expected": "Flagged",
        "invoice": """\
INVOICE
Invoice Number: INV-88291024
Date: 2026-09-08
Vendor: Shadow IT Consulting
Bill To: Enterprise Client Corp.

Line Items:
- Security Architecture Review: 0 units @ $750.00 each = $750.00
- Penetration Testing Report: 1 units @ $500.00 each = $500.00

Financial Summary:
Subtotal: $1250.00
Tax (10%): $125.00
Total: $1375.00""",
    },
]

# ── Helper Functions ──────────────────────────────────────────────────────────

def get_server_model_name(base_url):
    """Fetch the active model identifier registered in the vLLM server."""
    try:
        req = urllib.request.Request(f"{base_url}/models", headers={"User-Agent": "AuditLM-Client"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]["id"]
    except Exception:
        pass
    return DEFAULT_MODEL_NAME


def extract_verdict(text):
    """Extract verdict from the response."""
    tl = text.lower()
    # Check for XML tags or explicit markdown verdicts
    m_xml = re.search(r"<final_verdict>\s*(valid|flagged)\s*</final_verdict>", tl)
    if m_xml:
        return m_xml.group(1).capitalize()
    m = re.search(r"final\s+verdict:\s*(valid|flagged)", tl)
    if m:
        return m.group(1).capitalize()
    m2 = re.search(r"verdict:\s*(valid|flagged)", tl)
    if m2:
        return m2.group(1).capitalize()
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).lower()
    if "flagged" in body: return "Flagged"
    if "valid"   in body: return "Valid"
    return "Unknown"


def format_audit_display(full_text, elapsed):
    """Clean and display the reasoning trace and final verdict."""
    think_m = re.search(r"<think>(.*?)(?:</think>|$)", full_text, re.DOTALL | re.IGNORECASE)
    
    if think_m:
        think_text = think_m.group(1).strip()
        after_think = full_text[think_m.end():].strip()
    else:
        think_text = ""
        after_think = full_text.strip()

    # 1. Print Reasoning Trace
    if think_text:
        print()
        print(cyan(bold("  ┌─ 🧠 AUDIT REASONING TRACE (<think>) " + "─" * 38)))
        for line in think_text.splitlines():
            line_str = line.strip()
            if line_str:
                if re.match(r"^(step\s*\d+|[0-9]+\.|\*|\-)", line_str, re.IGNORECASE):
                    print(cyan(f"  │  {bold(line_str)}"))
                else:
                    print(dim(cyan(f"  │  {line_str}")))
            else:
                print(cyan("  │"))
        print(cyan(bold("  └" + "─" * 74)))

    # 2. Extract Verdict & Reason
    verdict = extract_verdict(full_text)
    reason_m = re.search(r"reason:\s*(.*)", after_think, re.IGNORECASE | re.DOTALL)
    reason = reason_m.group(1).strip() if reason_m else after_think

    # Clean up repetitive verdict strings and XML tags from reason
    reason = re.sub(r"<final_verdict>.*?</final_verdict>", "", reason, flags=re.IGNORECASE | re.DOTALL)
    reason = re.sub(r"</?reason>", "", reason, flags=re.IGNORECASE)
    reason = re.sub(r"^final verdict:\s*(valid|flagged)\s*", "", reason, flags=re.IGNORECASE).strip()
    # Remove any extra user/assistant turn artifacts
    reason = re.split(r"\n\s*(?:user|assistant|<\|im_start\|>|<think>)", reason, flags=re.IGNORECASE)[0].strip()

    print()
    if verdict == "Flagged":
        print(f"  {bold(BG_RED + '  🚩 VERDICT: FLAGGED (FRAUD / ANOMALY DETECTED)  ' + RESET)}")
    elif verdict == "Valid":
        print(f"  {bold(BG_GRN + '  ✅ VERDICT: VALID (PASSED COMPLIANCE)  ' + RESET)}")
    else:
        print(f"  {bold(yellow('  ❓ VERDICT: ' + verdict))}")

    print()
    if reason:
        print(f"  {bold('Reason:')}")
        for rline in reason.splitlines():
            if rline.strip():
                print(f"    {rline.strip()}")

    print(dim(f"\n  ⏱  Inference time: {elapsed:.2f}s | Tokens generated"))


def stream_from_vllm_api(base_url, model_name, invoice_text, max_tokens=512):
    """Stream response from vLLM's OpenAI-compatible /v1/chat/completions endpoint."""
    user_content = format_invoice_user_prompt(invoice_text)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop": ["<|im_end|>", "<|endoftext|>", "\nuser\n", "\nUser:", "\nAssistant:"],
        "stream": True,
    }

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "AuditLM-LiveClient"
        }
    )

    t0 = time.time()
    collected_tokens = []
    
    sys.stdout.write(yellow("  ⏳ Streaming reasoning from AuditLM...\r"))
    sys.stdout.flush()

    try:
        with urllib.request.urlopen(req) as response:
            sys.stdout.write(" " * 50 + "\r")  # clear spinner line
            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str == "data: [DONE]":
                    continue
                if line_str.startswith("data: "):
                    try:
                        chunk = json.loads(line_str[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected_tokens.append(content)
                    except json.JSONDecodeError:
                        continue
    except urllib.error.URLError as e:
        print(red(f"\n  [Error connecting to vLLM at {base_url}]: {e}"))
        print(yellow("  Make sure your vLLM server is running, e.g.:"))
        print(dim(f"  ollama serve   # then: ollama run {DEFAULT_MODEL_NAME}"))
        print(dim(f"  vllm serve {DEFAULT_GGUF_MODEL} --port 8000"))
        sys.exit(1)

    elapsed = time.time() - t0
    full_text = "".join(collected_tokens)
    return full_text, elapsed


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_single_audit(base_url, model_name, invoice_text, max_tokens=512):
    full_text, elapsed = stream_from_vllm_api(base_url, model_name, invoice_text, max_tokens)
    format_audit_display(full_text, elapsed)


def run_showcase(base_url, model_name, max_tokens=512):
    print()
    print(bold("=" * 76))
    print(bold("   🧾  AuditLM — Automated Showcase Test Cases"))
    print(bold("=" * 76))

    scores = []
    for idx, ex in enumerate(EXAMPLES, 1):
        print()
        print(bold(f"  ─── Case {idx}/{len(EXAMPLES)}: {ex['name']} ───"))
        print(dim("  [Input Invoice]"))
        for line in ex["invoice"].splitlines():
            print(dim(f"    {line}"))

        full_text, elapsed = stream_from_vllm_api(base_url, model_name, ex["invoice"], max_tokens)
        format_audit_display(full_text, elapsed)

        verdict = extract_verdict(full_text)
        is_correct = verdict.lower() == ex["expected"].lower()
        scores.append(is_correct)

        if is_correct:
            print(green(f"  ✓ EVALUATION PASSED (Expected: {ex['expected']}, Got: {verdict})"))
        else:
            print(red(f"  ✗ EVALUATION FAILED (Expected: {ex['expected']}, Got: {verdict})"))
        print(bold("─" * 76))

    passed = sum(scores)
    print()
    print(bold("=" * 76))
    summary_color = green if passed == len(EXAMPLES) else yellow
    print(summary_color(bold(f"  SHOWCASE RESULT: {passed}/{len(EXAMPLES)} Passed ({passed/len(EXAMPLES)*100:.0f}%)")))
    print(bold("=" * 76))
    print()


def run_interactive(base_url, model_name, max_tokens=512):
    print()
    print(bold("=" * 76))
    print(bold("   🧾  AuditLM — Interactive Invoice Auditor"))
    print(bold("=" * 76))
    print(dim("  Paste or type your invoice text below."))
    print(dim("  Press ENTER twice (or type 'audit') when done. Type 'exit' to quit.\n"))

    while True:
        print(cyan(bold("Invoice Input:")))
        lines = []
        try:
            while True:
                line = input()
                if line.strip().lower() in {"exit", "quit", "q"}:
                    print(dim("\n  Session closed. Goodbye!"))
                    return
                if line.strip().lower() == "audit":
                    break
                if line == "" and lines:
                    break
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            print(dim("\n  Session ended."))
            return

        invoice_content = "\n".join(lines).strip()
        if not invoice_content:
            continue

        run_single_audit(base_url, model_name, invoice_content, max_tokens)
        print()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AuditLM formatted vLLM client")
    parser.add_argument("--url", default=DEFAULT_SERVER_URL, help=f"vLLM server URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--model", default=None, help="Model name on server (auto-detected if omitted)")
    parser.add_argument("--example", action="store_true", help="Run 3 built-in showcase tests")
    parser.add_argument("--invoice", type=str, default=None, help="Single invoice text to audit directly")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens (default: 512)")
    args = parser.parse_args()

    # Autodetect model from server if not specified
    model_name = args.model or get_server_model_name(args.url)

    if args.example:
        run_showcase(args.url, model_name, args.max_tokens)
    elif args.invoice:
        run_single_audit(args.url, model_name, args.invoice, args.max_tokens)
    else:
        run_interactive(args.url, model_name, args.max_tokens)


if __name__ == "__main__":
    main()
