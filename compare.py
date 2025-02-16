import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.quantized.engine = 'qnnpack'

# 1. 加载原始模型和tokenizer
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"  # 根据实际路径修改
tokenizer = AutoTokenizer.from_pretrained(model_name)

original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 2. 准备量化配置（动态量化）
def quantize_model(model):
    # 动态量化主要量化Linear层
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model

# 3. 执行量化
quantized_model = quantize_model(original_model)

# 4. 测试和对比误差
prompt = "你是什么模型？"
inputs = tokenizer(prompt, return_tensors="pt").to(original_model.device)

# 获取原始模型输出
with torch.no_grad():
    original_outputs = original_model(**inputs)

# 获取量化模型输出
with torch.no_grad():
    quantized_outputs = quantized_model(**inputs)

# 5. 计算数值误差（MSE，L1误差）
original_logits = original_outputs.logits
quantized_logits = quantized_outputs.logits

# 均方误差 (MSE)
mse_error = F.mse_loss(quantized_logits, original_logits)

# 绝对误差 (L1)
l1_error = F.l1_loss(quantized_logits, original_logits)

print(f"均方误差 (MSE): {mse_error.item()}")
print(f"绝对误差 (L1): {l1_error.item()}")

# 6. 对比生成的文本
with torch.no_grad():
    original_text_outputs = original_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

with torch.no_grad():
    quantized_text_outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

# 打印原始模型和量化模型的生成结果
original_text = tokenizer.decode(original_text_outputs[0], skip_special_tokens=True)
quantized_text = tokenizer.decode(quantized_text_outputs[0], skip_special_tokens=True)

print(f"原始模型生成文本: {original_text}")
print(f"量化模型生成文本: {quantized_text}")
