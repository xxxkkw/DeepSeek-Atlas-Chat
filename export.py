import os
import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM

class QwenForCausalLMWrapper(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.args = args

    def forward(
            self,
            input_ids,
            attention_mask,
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        return outputs.logits


def export_qwen(args):

    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # 加载模型
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device, trust_remote_code=True, torch_dtype=dtype
    ).eval()

    print("Model loaded, starting export...")
    config = model.config
    qwen_model_wrapper = QwenForCausalLMWrapper(model, config, args)
    onnx_file_name = os.path.join(args.out_dir, "qwen_onnx.onnx")

    # 初始化输入数据
    batch = 1
    N = 1
    input_ids = torch.ones([batch, N], dtype=torch.int64)
    attention_mask = torch.ones([batch, N], dtype=torch.int64)

    input_names = ['input_ids', 'attention_mask']
    output_names = ['output']
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size', 1: 'sequence_length'}
    }

    print("开始导出 ONNX 模型...")
    torch.onnx.export(
        qwen_model_wrapper,

        (input_ids, attention_mask),
        onnx_file_name,
        opset_version=15,
        ir_version=7,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"ONNX 模型已导出至: {onnx_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export Qwen model to ONNX')
    parser.add_argument('-m', '--model_path', required=True, type=str, default="")
    parser.add_argument('-o', '--output_path', required=False, type=str, default="onnx_model_output")
    parser.add_argument('-d', '--device', required=False, type=str, choices=["mps", "cpu", "cuda"], default="cpu")
    parser.add_argument('-p', '--dtype', required=False, type=str, choices=["float32", "float16", "bfloat16"],
                        default="float32")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    export_qwen(args)