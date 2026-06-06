import tensorflow as tf
import numpy as np
import os
import requests
import csv

class YamnetClassifier:
    def __init__(self, model_dir="models/yamnet"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = self._load_or_download_model()
        self.class_map = self._load_class_map()
        print("✅ YAMNet 已准备好（本地模式）")
    
    def _load_or_download_model(self):
        saved_model_path = os.path.join(self.model_dir, "saved_model")

        if not os.path.exists(saved_model_path):
            raise RuntimeError(
                "❌ 未找到本地 YAMNet 模型，请确认 models/yamnet/saved_model 已上传到仓库"
            )

        print("✅ 从本地加载 YAMNet 模型")
        return tf.saved_model.load(saved_model_path)
    
    def _load_class_map(self):
        csv_path = os.path.join(self.model_dir, "yamnet_class_map.csv")
        
        if not os.path.exists(csv_path):
            print("正在下载类别映射表...")
            url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            response = requests.get(url)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("✅ 类别映射表下载完成")
        
        # 正确解析 CSV（支持带逗号的引号字段）
        class_names = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 3:
                    class_names.append(row[2])  # display_name
        
        print(f"✅ 共加载 {len(class_names)} 个音频类别")
        return np.array(class_names)
    
    def classify_from_file(self, audio_path):
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)
            return self.classify(audio, sr)
        except Exception as e:
            print("❌ 音频读取失败:", e)
            return []
    
    def classify(self, audio, sr=16000):
        if len(audio) < 16000:
            return []
        
        if sr != 16000:
            audio = tf.audio.resample(
                audio,
                rate_in=sr,
                rate_out=16000
            )
        
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _, _ = self.model(audio)
        scores_mean = np.mean(scores, axis=0)
        
        top5_idx = np.argsort(scores_mean)[-5:][::-1]
        results = [(self.class_map[i], float(scores_mean[i])) for i in top5_idx]
        return results