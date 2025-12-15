import numpy as np
from PIL import Image, ImageOps
from scipy.signal import istft
from .base import AudioEffect


class SpectrogramArtStyle(AudioEffect):
    """
    频谱画中音
    原理：利用 ISTFT 将图像的像素亮度映射为声音频谱的幅度。
    """

    def __init__(self, image_path, duration=5.0):
        super().__init__("Spectrogram Art Generator")
        self.image_path = image_path
        self.duration = duration
        self.n_fft = 2048
        self.hop_length = self.n_fft // 4

    def process(self, audio, samplerate):
        print(f"🎨 [SpectrogramArt] 正在尝试将图片 '{self.image_path}' 转换为音频...")

        try:
            # 1. 读取并处理图片
            img = Image.open(self.image_path).convert('L')

            # 2. 计算目标尺寸
            target_height = self.n_fft // 2 + 1
            target_width = int((self.duration * samplerate) / self.hop_length)

            # 3. 调整图片 (垂直翻转因为频谱图低频在下)
            img = img.resize((target_width, target_height), Image.Resampling.BICUBIC)
            img = ImageOps.flip(img)

            # 4. 构造频谱
            pixels = np.array(img) / 255.0
            random_phase = np.random.uniform(0, 2 * np.pi, pixels.shape)
            Zxx = (pixels ** 2) * np.exp(1j * random_phase)

            # 5. 逆变换生成音频
            _, generated_audio = istft(Zxx, fs=samplerate, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)

            # 6. 归一化
            max_val = np.max(np.abs(generated_audio))
            if max_val > 0:
                generated_audio = generated_audio / max_val * 0.95

            return generated_audio

        except Exception as e:
            print(f"❌ [Error] 无法读取图片: {e}")
            return np.zeros_like(audio)