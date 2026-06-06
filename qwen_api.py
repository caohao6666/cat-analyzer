import os
from dashscope import Generation

def analyze_cat_behavior(sound_description: str, image=None) -> str:
    """
    使用通义千问分析猫咪声音和行为
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return "❌ 未配置 DASHSCOPE_API_KEY"

    prompt = f"""
你是一名宠物行为学专家。

下面是对猫咪叫声的声音识别结果：
{sound_description}

请判断：
1. 猫咪可能的情绪状态
2. 当前行为含义
3. 主人应该如何应对

请用简体中文回答。
"""

    response = Generation.call(
        model="qwen-plus",
        prompt=prompt,
        api_key=api_key
    )

    if response.status_code != 200:
        return f"❌ 模型调用失败：{response.message}"

    return response.output.text