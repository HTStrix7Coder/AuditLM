# export_3b_gguf.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/models/auditlm_grpo_3b_final",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model.save_pretrained_gguf(
    "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/models/auditlm_3b_gguf",
    tokenizer,
    quantization_method = "q4_k_m"
)
print("✓ 3B GGUF export complete!")
