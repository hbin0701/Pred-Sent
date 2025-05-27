from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("/home/byeongguk/Latent_Step-1/src/sft/checkpoints/gsm8k_llama_instruct_code_rep/checkpoint-17184")  # LoRA 병합 후 저장된 모델 경로

# 실제 타입 확인
print(f"Model class: {type(model)}")
if isinstance(model, PeftModel):
    print("Lora Model")
else:
    print("Not Lora Model")
