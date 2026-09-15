import json
import os
import random
import re

def format_table_as_markdown(table_ori):
    if not table_ori or not isinstance(table_ori, list):
        return ""
    # Filter empty rows
    valid_rows = [row for row in table_ori if any(cell.strip() for cell in row)]
    if not valid_rows:
        return ""
    
    # Calculate column widths
    max_cols = max(len(row) for row in valid_rows)
    padded_rows = [row + [""] * (max_cols - len(row)) for row in valid_rows]
    
    lines = []
    # Header
    header = padded_rows[0]
    lines.append("| " + " | ".join(c.strip() for c in header) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in padded_rows[1:]:
        lines.append("| " + " | ".join(c.strip() for c in row) + " |")
    return "\n".join(lines)

def process_finqa(max_samples=1500):
    finqa_path = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets/FinQA/dataset/train.json"
    if not os.path.exists(finqa_path):
        print(f"FinQA not found at {finqa_path}")
        return []
    
    with open(finqa_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_samples = []
    for entry in data:
        qa = entry.get("qa", {})
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "")
        steps = qa.get("steps", [])
        gold_inds = qa.get("gold_inds", {})
        
        if not question or not answer:
            continue
            
        table_md = format_table_as_markdown(entry.get("table_ori", []))
        
        # Collect relevant text
        gold_sentences = list(gold_inds.values())
        relevant_text = "\n".join(gold_sentences) if gold_sentences else ""
        
        # Build prompt
        prompt = (
            "You are Audit-LM, an AI specialized in corporate finance and quantitative financial auditing.\n"
            "Analyze the provided financial data, extract the relevant metrics, and compute the answer.\n"
            "Think step-by-step inside <think>...</think> tags and provide your final numerical answer.\n\n"
        )
        if table_md:
            prompt += f"Financial Table:\n{table_md}\n\n"
        if relevant_text:
            prompt += f"Disclosures / Context:\n{relevant_text}\n\n"
        prompt += f"Question: {question}"
        
        # Build think steps
        think_lines = ["1. Identify relevant financial figures and formulas:"]
        if gold_sentences:
            for s in gold_sentences[:2]:
                think_lines.append(f"   - Reference: {s}")
        think_lines.append("2. Perform calculations:")
        
        if steps:
            for i, step in enumerate(steps, 1):
                op = step.get("op", "")
                arg1 = step.get("arg1", "")
                arg2 = step.get("arg2", "")
                res = step.get("res", "")
                think_lines.append(f"   - Step {i} ({op}): {arg1} and {arg2} -> Result = {res}")
        else:
            think_lines.append(f"   - Direct calculation yields: {answer}")
            
        think_lines.append(f"3. Final computation verified.")
        
        completion = "<think>\n" + "\n".join(think_lines) + f"\n</think>\n\nAnswer: {answer}"
        
        formatted_samples.append({
            "task_type": "financial_qa",
            "source": "FinQA",
            "prompt": prompt,
            "completion": completion,
            "ground_truth": answer,
            "metadata": {
                "question": question,
                "gold_steps": steps
            }
        })
        
        if len(formatted_samples) >= max_samples:
            break
            
    print(f"Processed {len(formatted_samples)} FinQA samples.")
    return formatted_samples

def process_finance_reasoning(max_samples=1000):
    fr_base = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets/FinanceReasoning/FinanceReasoning"
    files = ["easy.json", "medium.json"]
    
    formatted_samples = []
    for fname in files:
        fpath = os.path.join(fr_base, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8') as f:
            items = json.load(f)
            
        for item in items:
            q = item.get("question", "").strip()
            ctx = item.get("context", "")
            gt = item.get("ground_truth")
            sol = item.get("python_solution", "")
            
            if not q or gt is None:
                continue
                
            # Attempt to parse json context into clean table/text
            try:
                ctx_dict = json.loads(ctx) if isinstance(ctx, str) else ctx
                ctx_lines = []
                for k, v in ctx_dict.items():
                    ctx_lines.append(f"- {k}: {v}")
                formatted_ctx = "\n".join(ctx_lines)
            except Exception:
                formatted_ctx = str(ctx)
                
            prompt = (
                "You are Audit-LM, an AI specialized in financial compliance and quantitative reasoning.\n"
                "Analyze the following financial report data and solve the inquiry accurately.\n"
                "Think step-by-step inside <think>...</think> tags and provide your final calculated answer.\n\n"
                f"Financial Context:\n{formatted_ctx}\n\n"
                f"Question: {q}"
            )
            
            think_lines = [
                "1. Analyze Context and Question:",
                f"   - Target metric: {q}",
                "2. Extract values & execute logic:",
            ]
            if sol:
                for s_line in sol.strip().split("\n"):
                    if s_line.strip():
                        think_lines.append(f"   - {s_line.strip()}")
            think_lines.append(f"3. Calculated Ground Truth: {gt}")
            
            completion = "<think>\n" + "\n".join(think_lines) + f"\n</think>\n\nAnswer: {gt}"
            
            formatted_samples.append({
                "task_type": "financial_reasoning",
                "source": "FinanceReasoning",
                "prompt": prompt,
                "completion": completion,
                "ground_truth": str(gt),
                "metadata": {
                    "question": q,
                    "difficulty": item.get("level", "medium")
                }
            })
            
            if len(formatted_samples) >= max_samples:
                break
        if len(formatted_samples) >= max_samples:
            break
            
    print(f"Processed {len(formatted_samples)} FinanceReasoning samples.")
    return formatted_samples

def process_synthetic_fraud():
    fraud_path = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets/synthetic_fraud_7k.json"
    if not os.path.exists(fraud_path):
        print(f"Synthetic fraud dataset not found at {fraud_path}")
        return []
    with open(fraud_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    formatted_samples = []
    for item in items:
        formatted_samples.append({
            "task_type": "invoice_audit",
            "source": "synthetic_fraud_7k",
            "prompt": item["prompt"],
            "completion": item["completion"],
            "ground_truth": item["verdict"],
            "metadata": {
                "invoice_number": item["invoice_number"],
                "is_fraud": item["is_fraud"],
                "fraud_type": item["fraud_type"],
                "details": item["metadata"]
            }
        })
    print(f"Loaded {len(formatted_samples)} invoice audit samples.")
    return formatted_samples

def main():
    random.seed(42)
    
    print("--- Building Unified AuditLM Dataset ---")
    fraud_samples = process_synthetic_fraud()
    finqa_samples = process_finqa(max_samples=1500)
    fr_samples = process_finance_reasoning(max_samples=1000)
    
    all_samples = fraud_samples + finqa_samples + fr_samples
    random.shuffle(all_samples)
    
    total = len(all_samples)
    train_size = int(total * 0.9)
    train_samples = all_samples[:train_size]
    eval_samples = all_samples[train_size:]
    
    print(f"Total Unified Samples: {total}")
    print(f"Train Samples: {len(train_samples)}")
    print(f"Eval Samples: {len(eval_samples)}")
    
    output_dir = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets"
    train_out = os.path.join(output_dir, "auditlm_train.json")
    eval_out = os.path.join(output_dir, "auditlm_eval.json")
    
    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2)
    with open(eval_out, 'w', encoding='utf-8') as f:
        json.dump(eval_samples, f, indent=2)
        
    print(f"Successfully saved:")
    print(f" - Train set: {train_out}")
    print(f" - Eval set: {eval_out}")

if __name__ == "__main__":
    main()
