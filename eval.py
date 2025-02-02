import torch
import time
import onnxruntime
from transformers import AutoModelForCausalLM


pytorch_model_path = "./DeepSeek-R1-Distill-Qwen-1.5B"
onnx_model_path = "./deepseek_decoder.onnx"

# 初始化测试输入数据
batch_size = 1
seq_length = 10  # 测试序列长度
input_ids = torch.ones([batch_size, seq_length], dtype=torch.int64)
attention_mask = torch.ones([batch_size, seq_length], dtype=torch.int64)
position_ids = torch.arange(seq_length, dtype=torch.int64).unsqueeze(0)


def test_pytorch_model(model):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"PyTorch 模型平均推理时间: {avg_time:.4f} 秒")
    return outputs.logits.numpy()


def test_onnx_model(onnx_model_path):
    session = onnxruntime.InferenceSession(onnx_model_path)
    onnx_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    start_time = time.time()
    for _ in range(100):
        onnx_outputs = session.run(None, onnx_inputs)
    end_time = time.time()

    avg_time = (end_time - start_time) / 100
    print(f"ONNX 模型平均推理时间: {avg_time:.4f} 秒")

    return onnx_outputs[0]


pytorch_model = AutoModelForCausalLM.from_pretrained(pytorch_model_path).eval()


pytorch_logits = test_pytorch_model(pytorch_model)

# 测试 ONNX 模型
onnx_logits = test_onnx_model(onnx_model_path)

