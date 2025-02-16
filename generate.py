import random
from faker import Faker

fake = Faker("zh_CN")

# 1. 生成开放域聊天数据（60条）
structured_questions = [
    f"{fake.random_element(['推荐','如何','怎样','什么是','帮我','你知道','有没有'])}{fake.random_element(['北京适合亲子游的地方','缓解颈椎疼痛的方法','最近热门科幻电影','学习Python的技巧','咖啡对睡眠的影响','判断脱发的方法','量子纠缠的简单解释','去除衣服火锅味的小妙招','碳中和的民生影响'])}？" for _ in range(30)
]

free_text_questions = [
    f"{fake.sentence().replace('.','？')}" for _ in range(30)
]

chat_samples = structured_questions + free_text_questions

# 2. 生成家居控制指令（30条）
devices = ["客厅主灯", "卧室空调", "空气净化器", "扫地机器人", "窗帘",
          "书房台灯", "厨房电器", "电视", "智能插座", "暖风设备"]
actions = ["打开", "关闭", "调高", "调低", "查询", "启动", "设置", "调节"]

control_samples = [
    f"{random.choice(actions)}{random.choice(devices)}"
    + (f"至{random.randint(20,30)}度" if "调" in _ else "")
    for _ in range(20)
] + [
    f"{random.choice(['打开','关闭'])}所有{random.choice(['灯光','窗帘','电器'])}",
    f"将{random.choice(devices)}模式改为{random.choice(['静音','睡眠','强力'])}",
    "开启离家安防模式",
    "设置空调定时关闭（2小时后）"
]

# 3. 生成异常指令（10条）
error_samples = [
    f"{random.choice(['把','关'])}那个...呃...{random.choice(['灯','空调'])}{random.choice(['开','关'])}了吧",
    f"打开不存在的{random.choice(['地下室加湿器','花园喷泉'])}",
    f"设置空调温度为{random.choice(['-10','100'])}度",
    f"让{random.choice(['洗衣机','微波炉'])}{random.choice(['烤面包','播放音乐'])}",
    f"把{random.choice(['客厅','卧室'])}的...什么来着？",
    f"让空调进入{random.choice(['飞行','游戏'])}模式",
    f"关闭{random.choice(['厨房','阳台'])}的{random.choice(['智能马桶','魔法扫帚'])}"
]

# 4. 合并并打乱顺序
full_dataset = chat_samples + control_samples + error_samples
random.shuffle(full_dataset)

# 5. 保存到文件（每行一条）
with open("calibration_dataset.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(full_dataset[:100]))  # 确保总数100条