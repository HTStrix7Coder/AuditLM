import json
import os
import re
import torch
from unsloth import FastLanguageModel

from prompts import SYSTEM_PROMPT

def main():
    adapter_path = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/models/auditlm_grpo_final"
    eval_file = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets/auditlm_eval.json"
    
    print("=== Loading Audit-LM Fine-Tuned Model ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=False,
    )
    FastLanguageModel.for_inference(model)
    
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    print(f"\nLoaded {len(eval_data)} evaluation test cases.\n")
    
    # Select 3 interesting test cases (1 Valid invoice, 1 Fraud invoice, 1 Financial QA)
    test_cases = [
        eval_data[0], # Invoice audit
        eval_data[1], # Invoice audit
        eval_data[3], # Fraud invoice
        eval_data[8], # Financial calculation
    ]
    
    for idx, item in enumerate(test_cases, 1):
        print("="*70)
        print(f" TEST CASE #{idx} | Task: {item.get('task_type')} | Expected: {item.get('ground_truth')}")
        print("="*70)
        
        prompt = item["prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.2,
                top_p=0.9,
                use_cache=True
            )
            
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Parse think vs verdict
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if think_match:
            print("\n🧠 [MODEL THINKING TRACE / STEP-BY-STEP ARITHMETIC]:")
            print("-" * 50)
            print(think_match.group(1).strip())
            print("-" * 50)
            
            final_output = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            print("\n📋 [FINAL AUDIT VERDICT]:")
            print(final_output)
        else:
            print("\n📋 [RAW RESPONSE]:")
            print(response)
        print("\n")

if __name__ == "__main__":
    main()
