import json
import os
import re
import time
import torch

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_UNSLOTH = False

def extract_verdict(text):
    text_lower = text.lower()
    # Check for explicit Verdict: Valid/Flagged
    match = re.search(r"verdict:\s*(valid|flagged)", text_lower)
    if match:
        return match.group(1).capitalize()
    
    # Fallback search outside think tags
    content_after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).lower()
    if "verdict" in content_after_think:
        if "flagged" in content_after_think:
            return "Flagged"
        if "valid" in content_after_think:
            return "Valid"
    if "flagged" in content_after_think:
        return "Flagged"
    if "valid" in content_after_think:
        return "Valid"
    return "Unknown"

def extract_answer(text):
    # Search for Answer: ...
    match = re.search(r"answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback search for final line or last number
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return "Unknown"

def check_numeric_match(predicted, ground_truth, tolerance=0.05):
    # Extract floating numbers
    p_nums = re.findall(r"[-+]?\d*\.?\d+", str(predicted).replace(",", ""))
    g_nums = re.findall(r"[-+]?\d*\.?\d+", str(ground_truth).replace(",", ""))
    
    if not p_nums or not g_nums:
        return str(ground_truth).strip().lower() in str(predicted).strip().lower()
    
    try:
        p_val = float(p_nums[-1])
        g_val = float(g_nums[-1])
        if g_val == 0:
            return abs(p_val) < 1e-4
        return abs(p_val - g_val) / abs(g_val) <= tolerance
    except ValueError:
        return str(ground_truth).strip().lower() in str(predicted).strip().lower()

def main():
    model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    max_seq_length = 2048
    num_eval_samples = 40
    
    eval_file = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets/auditlm_eval.json"
    results_dir = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/eval_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"=== Audit-LM Baseline Evaluation ===")
    print(f"Loading model: {model_name}...")
    
    if HAS_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
        
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    selected_samples = eval_data[:num_eval_samples]
    print(f"Evaluating {len(selected_samples)} samples from test set...\n")
    
    correct_count = 0
    format_adherence_count = 0
    results_log = []
    
    start_time = time.time()
    
    for i, item in enumerate(selected_samples, 1):
        task_type = item.get("task_type", "unknown")
        prompt = item["prompt"]
        ground_truth = item["ground_truth"]
        
        messages = [
            {"role": "system", "content": "You are Audit-LM, an AI specialized in financial compliance, invoice verification, and quantitative audit reasoning."},
            {"role": "user", "content": prompt}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.9,
                use_cache=True
            )
            
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Check formatting (<think> tags)
        has_think_tags = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
        if has_think_tags:
            format_adherence_count += 1
            
        # Check accuracy based on task
        is_correct = False
        pred_val = ""
        if task_type == "invoice_audit":
            pred_verdict = extract_verdict(response)
            pred_val = pred_verdict
            is_correct = (pred_verdict.lower() == str(ground_truth).lower())
        else:
            pred_ans = extract_answer(response)
            pred_val = pred_ans
            is_correct = check_numeric_match(pred_ans, ground_truth)
            
        if is_correct:
            correct_count += 1
            
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"[{i:02d}/{len(selected_samples):02d}] Task: {task_type:<20} | Pred: {str(pred_val):<12} | GT: {str(ground_truth):<12} | Format: {'✓' if has_think_tags else '✗'} | {status}")
        
        results_log.append({
            "index": i,
            "task_type": task_type,
            "source": item.get("source", ""),
            "prompt": prompt,
            "ground_truth": ground_truth,
            "prediction": pred_val,
            "is_correct": is_correct,
            "has_think_tags": has_think_tags,
            "full_response": response
        })

    total_time = time.time() - start_time
    accuracy = (correct_count / len(selected_samples)) * 100
    format_rate = (format_adherence_count / len(selected_samples)) * 100
    
    summary = {
        "model": model_name,
        "total_evaluated": len(selected_samples),
        "correct_predictions": correct_count,
        "accuracy_percent": round(accuracy, 2),
        "format_adherence_percent": round(format_rate, 2),
        "total_time_seconds": round(total_time, 2),
        "details": results_log
    }
    
    output_path = os.path.join(results_dir, "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*50)
    print("           BASELINE EVALUATION RESULTS            ")
    print("="*50)
    print(f"Total Samples Tested  : {len(selected_samples)}")
    print(f"Overall Accuracy      : {accuracy:.2f}% ({correct_count}/{len(selected_samples)})")
    print(f"Format Adherence Rate : {format_rate:.2f}% ({format_adherence_count}/{len(selected_samples)})")
    print(f"Evaluation Runtime    : {total_time:.2f}s")
    print(f"Detailed Log Saved To : {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()
