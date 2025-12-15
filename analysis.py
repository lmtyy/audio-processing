import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.style.use('bmh')


class AudioAnalyzer:
    @staticmethod
    def calculate_snr(original, processed):
        """计算信噪比 (SNR)"""
        if len(original.shape) > 1: original = original[0]
        if len(processed.shape) > 1: processed = processed[0]

        min_len = min(len(original), len(processed))
        org = original[:min_len]
        proc = processed[:min_len]

        noise = org - proc
        p_signal = np.sum(org.astype(np.float64) ** 2)
        p_noise = np.sum(noise.astype(np.float64) ** 2)

        if p_noise < 1e-10: return float('inf')
        return 10 * np.log10(p_signal / p_noise)

    @staticmethod
    def plot_comparison(original, processed, samplerate, title="Analysis", filename="analysis.png"):
        """绘制频谱和波形对比图"""
        if len(original.shape) > 1: original = original[0]
        if len(processed.shape) > 1: processed = processed[0]

        min_len = min(len(original), len(processed))
        org = original[:min_len]
        proc = processed[:min_len]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # 1. 频谱对比
        def get_fft(y, sr):
            n = len(y)
            yf = fft(y)
            xf = fftfreq(n, 1 / sr)
            mask = (xf >= 0) & (xf <= sr / 2)
            return xf[mask], 2.0 / n * np.abs(yf[mask])

        x1, y1 = get_fft(org, samplerate)
        x2, y2 = get_fft(proc, samplerate)

        ax1 = axes[0]
        ax1.set_title(f"Spectrum: {title}")
        ax1.plot(x1, y1, 'g', alpha=0.5, label='Original')
        ax1.plot(x2, y2, 'r', alpha=0.6, label='Processed')
        ax1.legend()
        ax1.set_ylabel("Magnitude")

        # 2. 波形细节
        mid = len(org) // 2
        win = int(0.02 * samplerate)
        time_ax = np.linspace(0, 20, win)  # ms

        ax2 = axes[1]
        ax2.set_title("Waveform (20ms Zoom)")
        ax2.plot(time_ax, org[mid:mid + win], 'g--', alpha=0.5, label='Original')
        ax2.plot(time_ax, proc[mid:mid + win], 'b', alpha=0.8, label='Processed')
        ax2.set_xlabel("Time (ms)")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()
        print(f"📊 分析图表已生成: {filename}")