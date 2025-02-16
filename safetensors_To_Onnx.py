import os
import numpy as np
import torch
import onnx
from transformers import AutoModelForCausalLM, AutoTokenizer
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    CalibrationMethod,
    preprocess
)

# ================= 配置参数 =================
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
output_dir = "onnx_models_ort"
calibration_texts = [
    "你是谁？",
    "今天天气怎么样？",
    "请介绍一下自己。",
    "明天会下雨吗？",
    "如何制作巧克力蛋糕？",
    "中国的首都是哪里？",
    "1+1等于多少？",
    "推荐北京的美食",
    "怎样学习编程？",
    "讲个有趣的笑话",
    "Python和Java有什么区别？",
    "帮我写封感谢信",
    "量子计算是什么？",
    "如何保持健康？",
    "人类登月是哪年？",
    "解释相对论",
    "推荐周末活动",
    "怎样提高英语口语？",
    "巴黎铁塔多高？",
    "如何缓解压力？",
    "什么是人工智能？",
    "做披萨需要哪些材料？",
    "太阳系有几大行星？",
    "推荐经典科幻小说",
    "怎样快速入睡？",
    "解释区块链技术",
    "如何更换轮胎？",
    "恐龙为什么灭绝？",
    "写一首关于春天的诗",
    "咖啡对健康好吗？",
    "如何备份电脑文件？",
    "火星上有生命吗？",
    "推荐健身计划",
    "什么是碳中和？",
    "怎样种植多肉植物？",
    "比特币的工作原理",
    "如何拍摄星空？",
    "地球的年龄多大？",
    "解释机器学习",
    "推荐旅游目的地",
    "怎样训练狗狗？",
    "黑洞是什么？",
    "如何制作简历？",
    "水的沸点是多少？",
    "预防感冒的方法",
    "什么是元宇宙？",
    "推荐儿童读物",
    "怎样投资理财？",
    "金字塔是谁建的？",
    "解释光合作用",
    "如何清洗羊毛衫？",
    "宇宙有多大？",
    "推荐早餐食谱",
    "怎样缓解头痛？",
    "人工智能危险吗？",
    "如何解决失眠？",
    "世界上最长的河流",
    "推荐手机摄影技巧",
    "怎样学习日语？",
    "什么是5G技术？",
    "如何自制冰淇淋？",
    "大熊猫的习性",
    "推荐历史纪录片",
    "怎样处理焦虑？",
    "解释量子纠缠",
    "如何种植西红柿？",
    "地球到月球的距离",
    "推荐办公软件技巧",
    "怎样提高记忆力？",
    "什么是NFT？",
    "如何修理自行车？",
    "珠穆朗玛峰多高？",
    "推荐时间管理方法",
    "怎样保护眼睛？",
    "解释大数据技术",
    "如何腌制泡菜？",
    "火星适合居住吗？",
    "推荐亲子游戏",
    "怎样进行冥想？",
    "什么是深度学习？",
    "如何保养汽车？",
    "世界上最深的海沟",
    "推荐写作技巧",
    "怎样制作PPT？",
    "解释气候变化",
    "如何训练记忆力？",
    "恐龙有哪些种类？",
    "推荐家庭健身动作",
    "怎样制作动画？",
    "什么是云计算？",
    "如何折纸飞机？",
    "银河系有多大？",
    "推荐早餐搭配",
    "怎样预防中暑？",
    "解释基因编辑技术",
    "如何拍好短视频？",
    "世界上最古老的文明",
    "推荐读书方法",
    "怎样进行垃圾分类？",
    "什么是虚拟现实？",
    "如何制作太阳能灯？",
    "企鹅的生活习性",
    "推荐减脂食谱",
    "怎样提高注意力？",
]

# ================ 创建输出目录 ================
os.makedirs(output_dir, exist_ok=True)

# ================ 初始化组件 ================
tokenizer = AutoTokenizer.from_pretrained(model_name)
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)


# ============== 模型包装器定义 ==============
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
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

# ============== 生成动态示例输入 ================
dummy_input = tokenizer(
    "动态序列示例",
    return_tensors="pt",
    padding="max_length",  # 仅用于导出时确定形状
    max_length=64,  # 任意设置，实际使用时会动态调整
    truncation=True
)

# ============ 导出浮点ONNX模型 =============
float_onnx_path = os.path.join(output_dir, "model.onnx")
torch.onnx.export(
    wrapped_model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    float_onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14,
    verbose=True
)



# ========== 动态校准数据读取器 ============
class DynamicCalibrationReader(CalibrationDataReader):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        self.counter = 0

        # 自动确定最大长度
        self.actual_max_length = min(
            max([len(tokenizer.encode(t)) for t in texts]) + 2,
            max_length
        )

    def get_next(self):
        if self.counter >= len(self.texts):
            return None

        text = self.texts[self.counter]
        self.counter += 1

        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",  # 动态填充到实际最大长度
            max_length=self.actual_max_length,
            truncation=True
        )

        return {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }


# ============ 执行静态量化 ================
quantized_onnx_path = os.path.join(output_dir, "quantized_dynamic.onnx")

print("\n开始静态量化...")
quantize_static(
    model_input=float_onnx_path,  # 直接使用原始模型
    model_output=quantized_onnx_path,
    calibration_data_reader=DynamicCalibrationReader(
        tokenizer=tokenizer,
        texts=calibration_texts,
        max_length=512  # 设置足够大的最大长度
    ),
    quant_format=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.Entropy,
    extra_options={
        "ExtraSensitivity": {"Attention": 0.01},  # 增强注意力层的敏感度
        "ForceQuantizeNoInputCheck": True,
        "MatMulConstBOnly": False,
        #"AddQDQPairToWeight": True  # 允许动态量化权重
    }
)

print("量化完成！最终模型：", quantized_onnx_path)