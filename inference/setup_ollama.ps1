# Register AuditLM with Ollama (Windows PowerShell)
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Gguf = Join-Path $RepoRoot "models\auditlm_3b_gguf\auditlm-3b-q4_k_m.gguf"

if (-not (Test-Path $Gguf)) {
    Write-Host "ERROR: GGUF not found at:" -ForegroundColor Red
    Write-Host "  $Gguf"
    Write-Host "See models/README.md for download instructions."
    exit 1
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: ollama not installed. Get it from https://ollama.com/download" -ForegroundColor Red
    exit 1
}

Set-Location $RepoRoot
ollama create auditlm -f inference/Modelfile

Write-Host ""
Write-Host "Done. Try:" -ForegroundColor Green
Write-Host "  ollama run auditlm"
Write-Host "  python scripts/live_demo_vllm.py --url http://localhost:11434/v1 --model auditlm --example"
