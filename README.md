# DeepSeek R1 1.5B 模型在昇腾 Atlas 200I DK A2 上的推理项目

本项目基于 DeepSeek 最新的 R1 1.5B 模型，在昇腾 Atlas 200I DK A2 设备上进行推理，并使用 int8 量化技术将模型转换为 OM 模型。以下是项目的详细说明和使用指南。

---

## 项目简介

本项目旨在将 DeepSeek R1 1.5B 模型部署到昇腾 Atlas 200I DK A2 设备上，并通过 int8 量化技术优化模型推理性能。项目包含以下主要功能：

1. **模型导出**：将 DeepSeek R1 1.5B 模型导出为 ONNX 格式。
2. **模型量化**：将 ONNX 模型量化为 int8 格式。
3. **模型推理与测评**：在 Atlas 200I DK A2 设备上运行推理，并测评模型的推理速度和显存占用。

---

## 项目结构
```
DeepSeek-Atlas-Chat/
├── export.py # 将 DeepSeek R1 模型导出为 ONNX 格式
├── quant.py # 将 ONNX 模型量化为 int8 格式
├── eval.py # 测评模型推理速度和显存占用
├── onnx_model_output/ # 存放从原始 R1 模型导出的 ONNX 模型
├── deepseek_quant8.onnx # 自动生成的 int8 量化后的 ONNX 模型
└── README.md # 项目说明文档
```
---

## 环境要求

- **硬件**：昇腾 Atlas 200I DK A2，理论上也可以在其他310设备运行
- **软件**：
  - Python 3.7 或更高版本
  - ONNX 1.10.0 或更高版本
  - 昇腾 CANN 工具包（推荐版本 5.1.RC2）
  - PyTorch 1.8.0 或更高版本（用于模型导出）
  - ONNX Runtime（可选，用于本地测试）

---

## 使用步骤
### 1. 请先自行下载DeepSeek-R1-Distill-Qwen-1.5B模型

### 2. 导出 ONNX 模型

运行 `export.py` 脚本，将 DeepSeek R1 1.5B 模型导出为 ONNX 格式：

```bash
python export.py -m /path/to/DeepSeek-R1-Distill-Qwen-1.5B
```
若设备支持cuda或者mps，可在运行时加上`-d cuda`或者`-d mps`
### 3. 量化 ONNX 模型为 int8 格式

运行 quant.py 脚本，将 ONNX 模型量化为 int8 格式：

```bash
python quant.py
```
量化后的模型将保存为 `deepseek_quant8.onnx`

### 4. 测评模型推理性能
运行 eval.py 脚本，测评模型在 转换量化前后的推理速度：
```bash
python eval.py
```
得到结果大致如下，设备性能不同可能得到不同的结果，结果符合从fp32到int8的性能提升
```bash
测试 PyTorch 模型...
PyTorch 模型平均推理时间: 0.2651 秒
PyTorch 模型每秒处理 token 数量: 37.72 tokens/s

测试 ONNX 模型...
ONNX 模型平均推理时间: 0.0611 秒
ONNX 模型每秒处理 token 数量: 163.77 tokens/s

ONNX 模型相较于 PyTorch 模型推理速度提升: 4.34 倍
```
### 5. 转换 ONNX 模型为 OM 模型
使用昇腾 ATC 工具将量化后的 ONNX 模型转换为 OM 模型：  
```bash
atc --model=deepseek_onnx.onnx \
    --framework=5 \
    --output=deepseek_om_model \
    --input_format=ND \
    --input_shape="input_ids:-1,-1;attention_mask:-1,-1" \
    --dynamic_dims="1,64,1,64;8,128,8,128;16,256,16,256" \
    --soc_version=Ascend310B1 \
    --precision_mode=allow_fp32_to_fp16
```
这一步的时间会很长，耐心等待
