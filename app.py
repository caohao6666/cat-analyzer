import os
import time
import gradio as gr
import numpy as np
import soundfile as sf

from qwen_api import analyze_cat_behavior

# =========================
# 检查 API Key（启动即检查）
# =========================

if not os.environ.get("DASHSCOPE_API_KEY"):
    raise RuntimeError("❌ 未检测到 DASHSCOPE_API_KEY，请在 Hugging Face Secrets 中配置")

print("===== 应用程序启动于", time.strftime("%Y-%m-%d %H:%M:%S"), "=====")

# =========================
# 延迟加载 YAMNet（关键）
# =========================

yamnet_model = None

def get_yamnet_model():
    global yamnet_model
    if yamnet_model is None:
        print("⏳ 正在加载 YAMNet 模型（首次会比较慢）...")
        from yamnet_classifier import YamNetClassifier
        yamnet_model = YamNetClassifier()
        print("✅ YAMNet 模型加载完成")
    return yamnet_model

# =========================
# 核心分析函数（点击按钮才运行）
# =========================

def analyze(audio_file, image_file):
    if audio_file is None:
        return "❌ 请先上传音频文件"

    # 读取音频
    audio, sr = sf.read(audio_file)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 获取 YAMNet（此处才真正加载）
    yamnet = get_yamnet_model()

    # YAMNet 推理
    yamnet_results = yamnet.classify(audio, sr)

    # 整理成文本
    sound_desc = "\n".join(
        [f"{cls}: {score:.3f}" for cls, score in yamnet_results]
    )

    # 调用大模型分析（猫行为）
    result = analyze_cat_behavior(
        sound_description=sound_desc,
        image=image_file
    )

    return result

# =========================
# Gradio 界面
# =========================

with gr.Blocks(title="AI 猫咪语音识别分析") as demo:
    gr.Markdown(
        """
        # 🐱 AI 宠物语音识别分析  
        上传 **猫咪叫声音频**（可选图片），系统将分析猫咪的情绪与行为。
        """
    )

    with gr.Row():
        audio_input = gr.Audio(
            label="上传猫咪音频",
            type="filepath"
        )
        image_input = gr.Image(
            label="（可选）上传猫咪图片",
            type="pil"
        )

    analyze_btn = gr.Button("开始分析 🧠")
    output = gr.Textbox(
        label="分析结果",
        lines=10
    )

    analyze_btn.click(
        fn=analyze,
        inputs=[audio_input, image_input],
        outputs=output
    )

# =========================
# 启动
# =========================

demo.launch(server_name="0.0.0.0", server_port=7860)