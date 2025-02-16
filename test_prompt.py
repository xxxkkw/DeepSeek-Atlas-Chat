import torch
import numpy as np
import onnxruntime
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# 1. 加载模型与 Tokenizer
# -----------------------------
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载原始 PyTorch 模型（注意：如果你使用量化后的模型，请保证和ONNX版本一致）
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)
original_model.eval()

# 加载 ONNX 模型（例如使用 ONNX Runtime 后训练量化得到的模型）
onnx_model_path = "./onnx_models_ort/quantized_ort_model.onnx"  # 请替换为你的ONNX模型路径
ort_session = onnxruntime.InferenceSession(onnx_model_path)


# -----------------------------
# 2. 定义文本生成函数
# -----------------------------

# 使用 PyTorch 模型进行文本生成（调用 transformers 内置的 generate 方法）
def generate_text_pytorch(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated_ids = original_model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# 使用 ONNX 模型进行文本生成（采用贪婪解码实现简单的自回归生成）
def generate_text_onnx(prompt, max_new_tokens=50, eos_token_id=None):
    # 对 prompt 进行编码
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]  # shape: (1, seq_length)
    attention_mask = inputs["attention_mask"]

    # 自回归生成新 token
    for _ in range(max_new_tokens):
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        # 调用 ONNX 模型，输出 logits (形状为 (1, seq_length, vocab_size))
        ort_outs = ort_session.run(None, ort_inputs)
        logits = ort_outs[0]
        # 获取最后一个 token 的 logits
        last_logits = logits[0, -1, :]
        # 贪婪解码：选择概率最大的 token
        next_token_id = int(np.argmax(last_logits))
        # 若遇到 eos_token，则停止生成
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        # 将生成的 token 添加到输入序列中
        next_token_array = np.array([[next_token_id]], dtype=input_ids.dtype)
        input_ids = np.concatenate([input_ids, next_token_array], axis=1)
        # 更新 attention mask：新增 token 标记为1
        new_mask = np.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
        attention_mask = np.concatenate([attention_mask, new_mask], axis=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# -----------------------------
# 3. 生成并比较文本输出
# -----------------------------
prompt = "问题：你是谁？\n回答："

# 生成 PyTorch 模型的输出
pytorch_text = generate_text_pytorch(prompt, max_new_tokens=50)

# 获取 eos_token_id（如果 tokenizer 中定义了，则使用；否则为 None）
eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None

# 生成 ONNX 模型的输出
onnx_text = generate_text_onnx(prompt, max_new_tokens=50, eos_token_id=eos_token_id)

print("Prompt:", prompt)
print("-------------------------------------------------")
print("PyTorch 模型输出：")
print(pytorch_text)
print("-------------------------------------------------")
print("ONNX 模型输出：")
print(onnx_text)
