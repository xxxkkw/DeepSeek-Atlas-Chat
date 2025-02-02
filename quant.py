from onnxruntime.quantization import quantize_dynamic

if __name__ == '__main__':
    model_path = "./onnx_model_output/qwen_onnx.onnx"
    output_path = "./deepseek_decoder.onnx"

    quantize_dynamic(model_path, output_path)
    print("导出完成✅")
