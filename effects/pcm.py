import numpy as np
from .base import AudioEffect

class PCMBitcrusherStyle(AudioEffect):
    """
    [通信原理核心展示] 手写 PCM 量化
    模拟降低比特深度 (Bit Depth Reduction) 带来的量化噪声。
    从 16bit/32bit 降低到 4bit 或 8bit 风格。
    """
    def __init__(self, bit_depth=4):
        super().__init__(f"PCM Quantization ({bit_depth}-bit)")
        # 计算量化阶数，例如 4bit = 2^4 = 16 阶
        self.quantization_levels = 2 ** bit_depth

    def process(self, audio, samplerate):
        # 1. 归一化信号到 [0, 1] 区间以便计算
        # (假设输入 audio 范围是 -1 到 1)
        audio_normalized = (audio + 1.0) / 2.0
        
        # 2. 核心量化算法 (Quantization)
        # 将连续的模拟信号映射到离散的台阶上
        # y = floor(x * levels) / levels
        audio_quantized = np.floor(audio_normalized * self.quantization_levels) / self.quantization_levels
        
        # 3. 还原回 [-1, 1] 区间
        return audio_quantized * 2.0 - 1.0