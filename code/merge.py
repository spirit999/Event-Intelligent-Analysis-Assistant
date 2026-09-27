from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "/home/ps/wuzhichao/event_analysis/save_models/Qwen3-0.6B-KD-LoRA"
LORA_PATH = "/home/ps/wuzhichao/event_analysis/save_models/Qwen3-0.6B-KTO-LoRA"
SAVE_MERGED = "/home/ps/wuzhichao/event_analysis/save_models/Qwen3-0.6B-KTO"

config = PeftConfig.from_pretrained(LORA_PATH)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype="bfloat16",
    device_map="cpu",
)

lora_model = PeftModel.from_pretrained(model, LORA_PATH)

merged_model = lora_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
merged_model.save_pretrained(SAVE_MERGED)
tokenizer.save_pretrained(SAVE_MERGED)

print("LoRA合并完成！")
print(f"保存目录: {SAVE_MERGED}")
