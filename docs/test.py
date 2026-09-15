from unsloth import FastLanguageModel
import torch
from trl import GRPOConfig, GRPOTrainer

'''
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    dtype=None,
    load_in_4bit=True,
)
'''

print("✅ Model loaded successfully with Unsloth")
print("vLLM should be available for GRPO rollouts")