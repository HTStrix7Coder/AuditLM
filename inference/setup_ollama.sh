#!/usr/bin/env bash
# Register AuditLM with Ollama (Linux / macOS / WSL)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF="${REPO_ROOT}/models/auditlm_3b_gguf/auditlm-3b-q4_k_m.gguf"

if [[ ! -f "${GGUF}" ]]; then
  echo "ERROR: GGUF not found at:"
  echo "  ${GGUF}"
  echo "See models/README.md for download instructions."
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not installed. Get it from https://ollama.com/download"
  exit 1
fi

cd "${REPO_ROOT}"
ollama create auditlm -f inference/Modelfile
echo ""
echo "Done. Try:"
echo "  ollama run auditlm"
echo "  python scripts/live_demo_vllm.py --url http://localhost:11434/v1 --model auditlm --example"
