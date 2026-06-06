# 🐱 AI 宠物语音识别分析系统

一个基于 **YAMNet 音频分类** 和 **通义千问 AI 大模型** 的智能猫咪行为分析工具。上传猫咪叫声音频和图片，系统将智能识别猫咪的情绪状态和行为含义。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

---

## ✨ 功能特性

- 🎵 **音频分类** - 基于 YAMNet 模型进行声音识别和分类
- 🧠 **AI 行为分析** - 使用通义千问分析猫咪的情绪和行为
- 📸 **图像支持** - 可选上传猫咪图片作为辅助信息
- 🚀 **一键部署** - 支持 Docker 和 Hugging Face Spaces
- 💻 **Web 界面** - 基于 Gradio 的友好用户界面

---

## 🛠️ 技术栈

| 组件 | 说明 |
|------|------|
| **YAMNet** | Google 的音频事件识别模型 |
| **TensorFlow** | 深度学习框架 |
| **DashScope API** | 阿里云通义千问大模型 |
| **Gradio** | 快速构建 Web 界面 |
| **Python** | 核心开发语言 |

---

## 📋 项目结构
.
├── app.py                  # 主入口（Web 界面 / 服务）
├── yamnet_classifier.py    # YAMNet 音频分类逻辑
├── qwen_api.py             # 大模型分析接口
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 配置文件
├── models/
│   └── yamnet/                 # YAMNet 模型文件目录
│       ├── saved_model/        # TensorFlow SavedModel 格式
│       └── yamnet_class_map.csv # 音频类别映射表
└── README.md

---

## 🚀 快速开始

### 前置要求

- Python 3.8 或更高版本
- 阿里云通义千问 API Key（[获取链接](https://dashscope.console.aliyun.com/)）

### 本地安装

#### 1. 克隆仓库
```bash
git clone https://github.com/caohao6666/cat-analyzer.git
cd cat-analyzer

#### 2. 创建虚拟环境（推荐）
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

#### 3. 安装依赖
```bash
pip install -r requirements.txt

#### 4. 配置 API Key
```bash
# Linux/Mac
export DASHSCOPE_API_KEY="your_api_key_here"

# Windows (PowerShell)
$env:DASHSCOPE_API_KEY="your_api_key_here"

# Windows (CMD)
set DASHSCOPE_API_KEY=your_api_key_here

#### 5. 运行应用
```bash
python app.py

💻 使用方法
Web 界面操作
1.上传音频：在左侧输入框上传猫咪叫声音频文件（支持 .wav, .mp3 等格式）
2.上传图片（可选）：在右侧上传猫咪的图片，帮助 AI 更好地理解上下文
3.点击分析：点击「开始分析 🧠」按钮
4.查看结果：系统将返回：
---猫咪的情绪状态（开心、焦虑、警惕等）
---当前行为的含义
---主人应该如何应对的建议

📚 核心模块说明
1. yamnet_classifier.py - 音频识别引擎
功能：
---加载预训练的 YAMNet 模型（Google AudioSet）
---识别音频中的声音类别和特征
---返回置信度最高的 5 个类别
2. qwen_api.py - AI 行为分析
功能：
---调用阿里云通义千问 API
---将 YAMNet 识别结果转换为自然语言分析
---结合图片信息（如有）进行多模态理解
---提供宠物行为建议

🌐 云部署
部署到 HuggingFace Spaces
1.Fork 本仓库到 HuggingFace
2.创建新的 Space（选择 Docker 模板）
3.在 Space 的 Secrets 中添加：
---DASHSCOPE_API_KEY: 你的阿里云 API Key
4.上传 dockerfile 和所有源代码文件
5.Space 将自动构建并启动应用

使用 Docker 本地运行
```bash
# 构建镜像
docker build -t cat-analyzer .

# 运行容器
docker run -e DASHSCOPE_API_KEY="your_key" -p 7860:7860 cat-analyzer

⚙️ 配置和优化
性能优化
1.延迟加载：YAMNet 模型只在首次使用时加载，避免启动延迟
2.模型缓存：音频类别映射表本地保存，减少网络请求
3.批量处理：支持连续分析多个音频文件

自定义配置
修改 app.py 中的以下参数：
# Web 服务器配置
demo.launch(server_name="0.0.0.0", server_port=7860)

# 模型路径
model_dir = "models/yamnet"

# API 调用超时（qwen_api.py）
timeout = 30  # 秒

🙏 致谢
Google YAMNet - 音频分类模型
阿里云通义千问 - 大语言模型
Gradio - Web UI 框架
TensorFlow - 深度学习框架
