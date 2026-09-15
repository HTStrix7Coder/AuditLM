# Audit-LM: Verifiable Financial Reasoning Model with GRPO

## Project Overview
Audit-LM is a lightweight, verifiable LLM specialized in auditing financial documents (invoices, reports) for accuracy, consistency, and fraud detection. It uses Group Relative Policy Optimization (GRPO) to train step-by-step reasoning that is mathematically executable and auditable.

**Goal**: Transform black-box financial compliance into transparent, verifiable reasoning suitable for real-world auditing.

## Key Features
- Structured output: `<think>` reasoning + `Verdict: Valid/Flagged`
- Executable math verification
- Fraud pattern detection
- Lightweight (1.5B–3B parameters, 4-bit quantized)
- Local deployment with vLLM

## Project Planning (10 Weeks)

### Phase 1: Dataset Synthesis & Curation (Weeks 1–2)
- Collect FinQA + DocFinQA
- Generate synthetic invoices
- Inject realistic fraud cases (wrong totals, duplicates, phantom items, tax errors)
- Use teacher model to generate high-quality Chain-of-Thought traces

### Phase 2: Sandbox Setup & Baseline (Weeks 3–4)
- Set up Unsloth + vLLM environment
- Load base model (Qwen2.5-1.5B or Llama-3.2-3B)
- Evaluate baseline performance

### Phase 3: GRPO Training (Weeks 5–7)
- 4-bit QLoRA training
- Custom reward functions:
  - Format compliance
  - Math verification (executable accuracy)
  - Logical consistency
  - Verdict correctness

### Phase 4: Quantization, Testing & Serving (Weeks 8–9)
- Convert to GGUF 4-bit
- Comprehensive evaluation on test set
- Local serving with vLLM

### Phase 5: Documentation & Reporting (Week 10)
- Final report, logbook, presentation
- Performance benchmark with 8+ metrics

## Methodological Approach

### Data Pipeline
1. Raw data → Fraud injection scripts
2. Teacher model generates verifiable CoT
3. Hybrid dataset (public + synthetic)

### Training
- Base: Small efficient model
- Framework: Unsloth + GRPO
- Output format: Structured `<think>` + Verdict

### Evaluation Metrics
1. Pass@1
2. Maj@K (e.g. Maj@4)
3. Math Verification Accuracy (Execution Accuracy)
4. Fraud Detection F1-Score
5. Overall Binary Accuracy
6. Format Compliance
7. Average Completion Length
8. Logical Consistency Score

### Inference on New Unseen Data
- Model generates reasoning
- Automatic Python verifier runs arithmetic & structural checks
- Outputs Verdict + Verification Score
- Human review for low-confidence cases

## Expected Outcomes
- Significant improvement over base model in verifiable financial reasoning
- Production-ready lightweight model for invoice auditing
- Strong benchmark report demonstrating GRPO effectiveness

## Deliverables
- Trained model (GGUF)
- Training code & evaluation scripts
- Full project report + presentation
- Logbook with Gantt chart and weekly progress

---

**Project Status**: In Progress  
**Tech Stack**: Unsloth, vLLM, GRPO, Python, FinQA