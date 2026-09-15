import json
import os
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from prompts import SYSTEM_PROMPT

def load_sft_dataset(file_path, tokenizer, max_samples=None):
    """Load and format dataset for SFT training."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]

    formatted_texts = []
    for item in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)

    return Dataset.from_dict({"text": formatted_texts})


def main():
    # --- CONFIGURATION ---
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    max_seq_length = 1536
    output_dir = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/checkpoints/auditlm_sft_3b"
    train_file = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/datasets/auditlm_train.json"

    print("=== Audit-LM 3B SFT (Supervised Fine-Tuning Warmup) ===")
    print(f"Base Model     : {model_name}")
    print(f"Max Seq Length : {max_seq_length}")
    print(f"GPU Available  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )

    # 2. Add LoRA Adapters
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

    # 3. Load & Format Dataset
    print(f"\nLoading training data from {train_file}...")
    dataset = load_sft_dataset(train_file, tokenizer)
    print(f"Total training samples: {len(dataset)}")

    # 4. Configure SFT Training
    training_args = SFTConfig(
        output_dir=output_dir,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,    # Effective batch size = 8
        max_length=max_seq_length,
        max_steps=500,                    # ~12 minutes on RTX 4060 Ti
        packing=False,                    # Safe: does not require FlashAttention
        dataset_text_field="text",
        report_to="none",
    )

    # 5. Initialize SFTTrainer with Unsloth compatibility
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    # 6. Start SFT Training
    print("\nStarting SFT Training...")
    trainer.train()

    # 7. Save Model
    final_save_dir = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/models/auditlm_sft_3b_final"
    os.makedirs(final_save_dir, exist_ok=True)
    print(f"\nSaving SFT fine-tuned model to {final_save_dir}...")
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"SFT Training complete! Model saved to {final_save_dir}")


if __name__ == "__main__":
    main()
