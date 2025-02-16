from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == '__main__':
    model_path = "./onnx_model_output/qwen_onnx.onnx"
    output_path = "./deepseek_quant8.onnx"

    # 只对 Linear 层的权重进行量化，保持激活值和其他层为高精度
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,  # 只量化权重为 int8
        
    )
    print("量化模型导出完成✅")
