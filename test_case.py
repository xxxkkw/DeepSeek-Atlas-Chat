import torch
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.special import softmax  

# -----------------------------
# 1. 加载模型与 Tokenizer
# -----------------------------
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载原始（浮点或量化前）模型，这里以原始浮点模型为例
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)
original_model.eval()

# 如果你在导出时对模型进行了包装，这里也保持一致
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits

wrapped_model = ModelWrapper(original_model)
wrapped_model.eval()

# -----------------------------
# 2. 构造测试输入
# -----------------------------
test_text = "你是谁"
dummy_input = tokenizer(test_text, return_tensors="pt")
input_ids = dummy_input["input_ids"]
attention_mask = dummy_input["attention_mask"]

# -----------------------------
# 3. 使用 PyTorch 模型推理
# -----------------------------
with torch.no_grad():
    pt_logits = wrapped_model(input_ids, attention_mask)
    pt_logits = pt_logits.detach().cpu().numpy()  # shape: (batch_size, seq_len, vocab_size)

# -----------------------------
# 4. 使用 ONNX Runtime 推理
# -----------------------------
onnx_model_path = "./onnx_models_ort/quantized_ort_model.onnx"  # 请替换为你的 ONNX 模型路径
ort_session = onnxruntime.InferenceSession(onnx_model_path)
ort_inputs = {
    "input_ids": input_ids.cpu().numpy(),
    "attention_mask": attention_mask.cpu().numpy()
}
ort_outputs = ort_session.run(None, ort_inputs)
onnx_logits = ort_outputs[0]

# 检查形状是否一致
print("PyTorch logits shape:", pt_logits.shape)
print("ONNX logits shape:", onnx_logits.shape)

# -----------------------------
# 5. 计算 logits 的误差
# -----------------------------
abs_diff_logits = np.abs(pt_logits - onnx_logits)
max_diff_logits = np.max(abs_diff_logits)
mean_diff_logits = np.mean(abs_diff_logits)

print("【Logits】")
print("最大绝对误差：", max_diff_logits)
print("均值绝对误差：", mean_diff_logits)

# -----------------------------
# 6. 计算 softmax 概率分布的误差
# -----------------------------
# 注意：这里对 logits 最后一个维度（vocab维度）做 softmax 计算
pt_probs = softmax(pt_logits, axis=-1)    # shape 与 logits 相同
onnx_probs = softmax(onnx_logits, axis=-1)

abs_diff_probs = np.abs(pt_probs - onnx_probs)
max_diff_probs = np.max(abs_diff_probs)
mean_diff_probs = np.mean(abs_diff_probs)

print("【Softmax 概率分布】")
print("最大绝对误差：", max_diff_probs)
print("均值绝对误差：", mean_diff_probs)

# -----------------------------
# 7. 绘制直方图比较
# -----------------------------
plt.figure(figsize=(12, 5))

# logits 误差直方图
plt.subplot(1, 2, 1)
plt.hist(abs_diff_logits.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Logits 绝对误差直方图")
plt.xlabel("绝对误差")
plt.ylabel("频数")

# softmax 概率误差直方图
plt.subplot(1, 2, 2)
plt.hist(abs_diff_probs.flatten(), bins=50, color='green', alpha=0.7)
plt.title("Softmax 概率绝对误差直方图")
plt.xlabel("绝对误差")
plt.ylabel("频数")

plt.tight_layout()
plt.show()

# -----------------------------
# 8. 打印部分样本对比
# -----------------------------
print("\n样本对比：")
print("PyTorch logits 样本：", pt_logits.flatten()[:10])
print("ONNX logits 样本：", onnx_logits.flatten()[:10])
print("PyTorch softmax 样本：", pt_probs.flatten()[:10])
print("ONNX softmax 样本：", onnx_probs.flatten()[:10])