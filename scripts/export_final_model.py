import os
import shutil

src = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/checkpoints/auditlm_grpo/checkpoint-300"
dst = "/home/harinderan/AI_Projects/Unsloth_Finetuning/AuditLM/models/auditlm_grpo_final"

os.makedirs(dst, exist_ok=True)
for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)
        
print(f"Successfully exported final fine-tuned model to: {dst}")
print("Files:", os.listdir(dst))
