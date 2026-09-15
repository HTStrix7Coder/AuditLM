import json
import os
import re
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

from prompts import SYSTEM_PROMPT

# --- UTILITY TO SAFELY PARSE COMPLETIONS ---

def extract_text(completion):
    """
    Safely extract string content from raw string, dict, or list of message dicts.
    """
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
            elif isinstance(msg, str):
                return msg
        if completion and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return " ".join(str(x) for x in completion)
    elif isinstance(completion, dict):
        return completion.get("content", str(completion))
    return str(completion)

# --- STRICT VERIFIABLE REWARD FUNCTIONS FOR AUDIT-LM GRPO ---

def format_reward_func(prompts, completions, **kwargs):
    """
    Rewards proper <think>...</think> structure (+1.0).
    Penalizes missing or malformed tags (-1.0).
    Penalizes lazy or empty reasoning inside <think> (-0.5 to -1.0).
    """
    rewards = []
    for comp in completions:
        text = extract_text(comp)
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            words = think_content.split()
            digits = re.findall(r"\d", think_content)
            # Must contain actual numbers and enough reasoning depth
            if len(words) >= 25 and len(digits) >= 4:
                rewards.append(1.0)
            elif len(words) >= 10:
                rewards.append(0.2)
            else:
                rewards.append(-0.5)  # too short/lazy reasoning
        elif "<think>" in text or "</think>" in text:
            rewards.append(-0.5)  # unclosed or malformed
        else:
            rewards.append(-1.0)  # completely missing think tags
    return rewards

def structure_reward_func(prompts, completions, **kwargs):
    """
    Rewards structured conclusions:
    - For invoice_audit: 'Verdict: Valid' or 'Verdict: Flagged' (+1.0)
    - For financial_qa: 'Answer: ...' (+1.0)
    Penalizes missing verdicts (-0.5).
    """
    task_types = kwargs.get("task_type", ["invoice_audit"] * len(completions))
    rewards = []
    for comp, t_type in zip(completions, task_types):
        text = extract_text(comp).lower()
        if t_type == "invoice_audit":
            if "verdict: valid" in text or "verdict: flagged" in text:
                rewards.append(1.0)
            elif "verdict:" in text:
                rewards.append(0.2)
            else:
                rewards.append(-0.5)
        else:
            if "answer:" in text:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
    return rewards

def arithmetic_verification_reward_func(prompts, completions, **kwargs):
    """
    STRICT ARITHMETIC REWARD:
    Inspects inside <think>...</think> to verify that the model actually performed
    real mathematical verification operations (e.g. 10 x $156.31 = $1563.10, subtotals, tax sums).
    Penalizes lazy/vague reasoning with NO explicit arithmetic (-1.0).
    """
    rewards = []
    for comp in completions:
        text = extract_text(comp)
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if not think_match:
            rewards.append(-1.0)
            continue
            
        think_text = think_match.group(1)
        
        # Explicit math operations: qty x price = total or addition sums
        math_ops = re.findall(
            r"\d+(?:\.\d+)?\s*(?:[xX*+\-]|and)\s*\$?\d+(?:\.\d+)?\s*(?:=|->|=>|results? in)\s*\$?\d+(?:\.\d+)?",
            think_text
        )
        
        # Subtotal/tax calculations: "15% on $4475.06 = $671.26" or "$4475.06 + $671.26 = $5146.32"
        tax_or_sum_checks = re.findall(
            r"(?:\d+%\s*(?:of|on)\s*\$?\d+(?:\.\d+)?|\$?\d+(?:\.\d+)?\s*\+\s*\$?\d+(?:\.\d+)?\s*=)",
            think_text,
            flags=re.IGNORECASE
        )
        
        total_ops = len(math_ops) + len(tax_or_sum_checks)
        
        if total_ops >= 3:
            rewards.append(2.0)   # High reward: computed 3+ line items/sums explicitly
        elif total_ops >= 1:
            rewards.append(1.0)   # Moderate reward: computed 1-2 operations
        else:
            # Check if basic arithmetic syntax exists with digits
            has_basic_math = bool(re.search(r"\d+\s*[xX*+\-]\s*\$?\d+", think_text))
            if has_basic_math:
                rewards.append(0.2)
            else:
                # Lazy shortcut penalty: wrote words like "Verify total" with zero actual math
                rewards.append(-1.0)
    return rewards

def reasoning_consistency_reward_func(prompts, completions, **kwargs):
    """
    Ensures logical consistency between the reasoning trace and final verdict.
    - If reasoning detects discrepancy/mismatch/error/!= -> Verdict MUST be Flagged.
    - If reasoning verifies all calculations match/correct -> Verdict MUST be Valid.
    - Penalizes contradictions (-1.0) to eliminate blind guessing.
    """
    task_types = kwargs.get("task_type", ["invoice_audit"] * len(completions))
    rewards = []
    for comp, t_type in zip(completions, task_types):
        if t_type != "invoice_audit":
            rewards.append(0.0)
            continue
            
        text = extract_text(comp)
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue
            
        think_lower = think_match.group(1).lower()
        verdict_valid = "verdict: valid" in text.lower()
        verdict_flagged = "verdict: flagged" in text.lower()
        
        found_error = any(term in think_lower for term in [
            "discrepancy", "incorrect", "mismatch", "error", "≠", "!=", "does not match", "fraud"
        ])
        found_sound = any(term in think_lower for term in [
            "all correct", "matches stated", "verified and consistent", "no discrepancies", "✓ correct"
        ])
        
        if found_error and verdict_flagged:
            rewards.append(1.0)
        elif found_sound and not found_error and verdict_valid:
            rewards.append(1.0)
        elif (found_error and verdict_valid) or (found_sound and not found_error and verdict_flagged):
            rewards.append(-1.0)  # Direct contradiction penalty
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(prompts, completions, **kwargs):
    """
    High-value ground truth reward (+3.0) for correct audit verdict or numerical calculation.
    Penalizes wrong verdicts (-1.5) to give strong discriminative advantage in GRPO.
    """
    ground_truths = kwargs.get("ground_truth", [""] * len(completions))
    task_types = kwargs.get("task_type", ["invoice_audit"] * len(completions))
    rewards = []
    
    for comp, gt, t_type in zip(completions, ground_truths, task_types):
        text = extract_text(comp)
        gt_str = str(gt).strip().lower()
        
        if t_type == "invoice_audit":
            content_after = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).lower()
            pred = "unknown"
            if "verdict: valid" in content_after or "verdict: valid" in text.lower():
                pred = "valid"
            elif "verdict: flagged" in content_after or "verdict: flagged" in text.lower():
                pred = "flagged"
            elif "flagged" in content_after:
                pred = "flagged"
            elif "valid" in content_after:
                pred = "valid"
                
            if pred == gt_str:
                rewards.append(3.0)
            elif pred == "unknown":
                rewards.append(-0.5)
            else:
                rewards.append(-1.5)  # Penalize wrong verdict
        else:
            match = re.search(r"answer:\s*([^\n]+)", text, flags=re.IGNORECASE)
            pred_ans = match.group(1).strip() if match else text.strip().split("\n")[-1]
            
            p_nums = re.findall(r"[-+]?\d*\.?\d+", pred_ans.replace(",", ""))
            g_nums = re.findall(r"[-+]?\d*\.?\d+", gt_str.replace(",", ""))
            
            is_match = False
            if p_nums and g_nums:
                try:
                    p_val = float(p_nums[-1])
                    g_val = float(g_nums[-1])
                    if g_val == 0:
                        is_match = abs(p_val) < 1e-4
                    else:
                        is_match = (abs(p_val - g_val) / abs(g_val)) <= 0.05
                except ValueError:
                    is_match = gt_str in pred_ans.lower()
            else:
                is_match = gt_str in pred_ans.lower()
                
            if is_match:
                rewards.append(3.0)
            else:
                rewards.append(-1.5)
            
    return rewards

def load_audit_dataset(file_path, max_samples=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]

    formatted_rows = {
        "prompt": [],
        "ground_truth": [],
        "task_type": []
    }
    
    for item in data:
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]}
        ]
        formatted_rows["prompt"].append(prompt_msgs)
        formatted_rows["ground_truth"].append(str(item["ground_truth"]))
        formatted_rows["task_type"].append(item.get("task_type", "invoice_audit"))
        
    return Dataset.from_dict(formatted_rows)

def main():
    # --- CONFIGURATION ---
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    max_seq_length = 1536
    output_dir = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/checkpoints/auditlm_grpo_3b"
    train_file = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets/auditlm_train.json"
    
    print("=== Initializing Audit-LM 3B GRPO Reinforcement Learning (Strict Math Rewards) ===")
    print(f"Base Model     : {model_name}")
    print(f"Max Seq Length : {max_seq_length}")
    print(f"GPU Available  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 1. Load Model & Tokenizer (Pure GRPO from Base Model)
    load_source = model_name
    print(f"--> Initializing GRPO directly from base model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_source,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )
    
    # 2. Add or resume LoRA Adapters
    if not hasattr(model, "peft_config"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    
    # 3. Load Dataset
    print(f"\nLoading training data from {train_file}...")
    dataset = load_audit_dataset(train_file)
    print(f"Total training samples: {len(dataset)}")
    
    # 4. Configure GRPO Training Parameters with Extended Budget & Strict Verification
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,               # Stable RL policy gradient updates
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_steps=25,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        per_device_train_batch_size=1,    # 1 prompt per step
        gradient_accumulation_steps=1,
        num_generations=4,                # 4 exploration candidate outputs per prompt
        max_prompt_length=512,
        max_completion_length=384,        # Room for full arithmetic verification traces
        max_steps=1000,                   # 1000 steps = 4000 rollouts for true reasoning discovery
        dataloader_num_workers=2,
        report_to="none",
        use_vllm=False,
    )
    
    # 5. Initialize GRPOTrainer with Anti-Hack & Strict Math Reward Functions
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            structure_reward_func,
            arithmetic_verification_reward_func,
            reasoning_consistency_reward_func,
            accuracy_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    # 6. Start GRPO Training
    print("\nStarting GRPO Training (Strict Math & Anti-Shortcut Rewards)...")
    trainer.train()
    
    # 7. Save the Fine-Tuned Model & Adapters
    final_save_dir = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/models/auditlm_grpo_3b_final"
    os.makedirs(final_save_dir, exist_ok=True)
    print(f"\nSaving final 3B model LoRA adapters and tokenizer to {final_save_dir}...")
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Training complete! Final 3B model saved to {final_save_dir}")

if __name__ == "__main__":
    main()
