import torch
import time
import onnxruntime
from transformers import AutoModelForCausalLM

# 模型路径
pytorch_model_path = "./DeepSeek-R1-Distill-Qwen-1.5B"
onnx_model_path = "./deepseek_quant8.onnx"

# 初始化测试输入数据
batch_size = 1
seq_length = 10
input_ids = torch.ones([batch_size, seq_length], dtype=torch.int64)
attention_mask = torch.ones([batch_size, seq_length], dtype=torch.int64)
position_ids = torch.arange(seq_length, dtype=torch.int64).unsqueeze(0)


def test_pytorch_model(model):
    model.eval()
    with torch.no_grad():
        # 推理测试
        start_time = time.time()
        for i in range(100):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        end_time = time.time()

    # 计算平均推理时间和每秒处理 token 数量
    avg_time = (end_time - start_time) / 100
    total_tokens = batch_size * seq_length * 100  # 总 token 数量
    tokens_per_second = total_tokens / (end_time - start_time)

    print(f"PyTorch 模型平均推理时间: {avg_time:.4f} 秒")
    print(f"PyTorch 模型每秒处理 token 数量: {tokens_per_second:.2f} tokens/s")

    # 将输出张量移动到 CPU 并转换为 NumPy 数组
    return outputs.logits.cpu().numpy(), avg_time


def test_onnx_model(onnx_model_path):
    session = onnxruntime.InferenceSession(onnx_model_path)
    onnx_inputs = {
        "input_ids": input_ids.cpu().numpy(),  # 先移回 CPU 再转换为 NumPy 数组
        "attention_mask": attention_mask.cpu().numpy(),  # 同样处理 attention_mask
    }

    # 推理测试
    start_time = time.time()
    for i in range(100):
        onnx_outputs = session.run(None, onnx_inputs)
    end_time = time.time()

    # 计算平均推理时间和每秒处理 token 数量
    avg_time = (end_time - start_time) / 100
    total_tokens = batch_size * seq_length * 100  # 总 token 数量
    tokens_per_second = total_tokens / (end_time - start_time)

    print(f"ONNX 模型平均推理时间: {avg_time:.4f} 秒")
    print(f"ONNX 模型每秒处理 token 数量: {tokens_per_second:.2f} tokens/s")

    return onnx_outputs[0], avg_time


# 使用 CPU 设备
device = torch.device("cpu")
print(f"使用设备: {device}")

# 加载 PyTorch 模型
pytorch_model = AutoModelForCausalLM.from_pretrained(pytorch_model_path).eval()

# 将模型和输入数据移动到 CPU
pytorch_model.to(device)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
position_ids = position_ids.to(device)

# 测试 PyTorch 模型
print("测试 PyTorch 模型...")
pytorch_logits, pytorch_avg_time = test_pytorch_model(pytorch_model)

# 测试 ONNX 模型
print("\n测试 ONNX 模型...")
onnx_logits, onnx_avg_time = test_onnx_model(onnx_model_path)

# 计算推理速度提升倍数
speedup = pytorch_avg_time / onnx_avg_time
print(f"\nONNX 模型相较于 PyTorch 模型推理速度提升: {speedup:.2f} 倍")
