import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.backends.quantized.engine = 'qnnpack'
# 1. 加载原始模型和tokenizer
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"  # 根据实际路径修改
tokenizer = AutoTokenizer.from_pretrained(model_name)
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # 量化需要float32格式
    device_map="cpu"
)


# 2. 准备量化配置（动态量化）
def quantize_model(model):
    # 动态量化主要量化Linear层
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 指定要量化的层类型
        dtype=torch.qint8
    )
    return quantized_model


# 3. 执行量化
quantized_model = quantize_model(original_model)

# 4. 测试量化后的模型
prompt = "你是谁"
inputs = tokenizer(prompt, return_tensors="pt").to(quantized_model.device)

with torch.no_grad():
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# 5. 保存量化模型（需要自定义处理）
def save_quantized_model(model, save_path):
    # PyTorch动态量化模型需要特殊保存方式
    torch.save(model.state_dict(), save_path)


save_path = "./DeepSeek-R1-Distill-Qwen-1.5B_int8.safetensors"
save_quantized_model(quantized_model, save_path)
print("保存成功")