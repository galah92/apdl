import warnings
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import pyworld as pw  # type: ignore[import-untyped]  # noqa: E402

plt.rcParams["savefig.dpi"] = 150


def plot_audio_analysis(
    audio: np.ndarray,
    sr: int,
    title: str,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> plt.Figure:
    """Plot waveform, spectrogram with pitch, mel-spectrogram, and energy/RMS."""
    n_fft = int(window_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    duration = len(audio) / sr
    time = np.arange(len(audio)) / sr

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 12), sharex=True, constrained_layout=True
    )
    fig.suptitle(title, fontsize=14)

    # Waveform
    axes[0].plot(time, audio, linewidth=0.5)
    axes[0].set(ylabel="Amplitude", title="Waveform", xlim=[0, duration])

    # Spectrogram
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=axes[1]
    )
    axes[1].set(ylabel="Frequency [Hz]", title=f"Spectrogram (F_max = {sr // 2} Hz)")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    # Pitch contour
    f0, t_f0 = pw.harvest(audio.astype(np.float64), sr, frame_period=hop_ms)  # type: ignore[attr-defined]
    voiced = f0 > 0
    axes[1].scatter(t_f0[voiced], f0[voiced], c="cyan", s=2, alpha=0.8, label="F0")
    axes[1].legend(loc="upper right")

    # Mel-spectrogram (n_mels must be <= n_fft//2 to avoid empty filters)
    n_mels = min(128, n_fft // 4)
    S_mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    img_mel = librosa.display.specshow(
        librosa.power_to_db(S_mel, ref=np.max),
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=axes[2],
    )
    axes[2].set(ylabel="Frequency [Hz]", title="Mel-Spectrogram")
    fig.colorbar(img_mel, ax=axes[2], format="%+2.0f dB")

    # Energy and RMS
    frames = librosa.util.frame(audio, frame_length=n_fft, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    n_frames = min(len(energy), len(rms))
    energy, rms = energy[:n_frames], rms[:n_frames]
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length
    )

    ax_rms = axes[3].twinx()
    l1 = axes[3].plot(frame_times, energy, "b-", linewidth=0.8, label="Energy")
    l2 = ax_rms.plot(frame_times, rms, "r-", linewidth=0.8, label="RMS")
    axes[3].set(xlabel="Time [sec]", ylabel="Energy", title="Energy and RMS")
    axes[3].tick_params(axis="y", labelcolor="b")
    ax_rms.set_ylabel("RMS", color="r")
    ax_rms.tick_params(axis="y", labelcolor="r")
    axes[3].legend(l1 + l2, ["Energy", "RMS"], loc="upper right")

    return fig


def plot_signals(
    signals: list[tuple[np.ndarray, str]], sr: int, title: str
) -> plt.Figure:
    """Plot multiple signals vertically."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 3 * len(signals)))
    fig.suptitle(title, fontsize=14)
    if len(signals) == 1:
        axes = [axes]
    for ax, (sig, name) in zip(axes, signals):
        time = np.arange(len(sig)) / sr
        ax.plot(time, sig, linewidth=0.5)
        ax.set(xlabel="Time [sec]", ylabel="Amplitude", title=name)
    plt.tight_layout()
    return fig


def spectral_subtraction(
    audio: np.ndarray,
    sr: int,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
    vad_threshold: float = 0.1,
    alpha: float = 2.0,
    beta: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply spectral subtraction for noise reduction."""
    n_fft = int(window_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)

    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(D), np.angle(D)

    # VAD based on energy (use magnitude sum to match STFT frame count)
    energy = np.sum(mag**2, axis=0)
    threshold = vad_threshold * np.median(energy[energy > np.percentile(energy, 10)])
    is_speech = energy > threshold

    # Initialize noise estimate from non-speech frames
    non_speech = np.where(~is_speech[:10])[0]
    noise_est = (
        np.mean(mag[:, non_speech], axis=1)
        if len(non_speech) > 0
        else np.mean(mag[:, :5], axis=1)
    )

    # Process frames sequentially
    enhanced_mag = np.zeros_like(mag)
    for i in range(mag.shape[1]):
        if not is_speech[i]:
            noise_est = 0.9 * noise_est + 0.1 * mag[:, i]
        enhanced_mag[:, i] = np.maximum(mag[:, i] - alpha * noise_est, beta * mag[:, i])

    enhanced = librosa.istft(
        enhanced_mag * np.exp(1j * phase), hop_length=hop_length, length=len(audio)
    )
    return enhanced.astype(np.float32), energy, threshold


def plot_vad(
    energy: np.ndarray, threshold: float, sr: int, hop_length: int
) -> plt.Figure:
    """Plot energy with VAD threshold."""
    fig, ax = plt.subplots(figsize=(14, 4))
    t = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    ax.plot(t, energy, label="Energy")
    ax.axhline(
        threshold, color="r", linestyle="--", label=f"Threshold = {threshold:.4f}"
    )
    ax.set(xlabel="Time [sec]", ylabel="Energy", title="VAD Energy Threshold")
    ax.legend()
    plt.tight_layout()
    return fig


def auto_gain_control(
    audio: np.ndarray,
    sr: int,
    target_db: float = -20.0,
    noise_floor_db: float = -50.0,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
    stats_window_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply AGC to normalize audio levels."""
    n_fft = int(window_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    stats_frames = int(stats_window_sec * sr / hop_length)

    target_rms = 10 ** (target_db / 20)
    noise_floor = 10 ** (noise_floor_db / 20)

    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    gains = np.ones_like(rms)

    for i in range(len(rms)):
        local_rms = np.mean(
            rms[max(0, i - stats_frames // 2) : min(len(rms), i + stats_frames // 2)]
        )
        if local_rms > noise_floor:
            gains[i] = target_rms / (local_rms + 1e-10)

    gains = scipy.signal.medfilt(gains, kernel_size=5)

    # Overlap-add synthesis
    frames = librosa.util.frame(audio, frame_length=n_fft, hop_length=hop_length)
    window = scipy.signal.windows.hann(n_fft)
    output = np.zeros(len(audio))
    norm = np.zeros(len(audio))

    for i in range(frames.shape[1]):
        start, end = i * hop_length, i * hop_length + n_fft
        if end > len(audio):
            break
        output[start:end] += frames[:, i] * gains[i] * window
        norm[start:end] += window

    norm[norm < 1e-10] = 1e-10
    output = np.tanh(output / norm / 0.9) * 0.9  # Soft clip
    return output.astype(np.float32), gains


def plot_gains(gains: np.ndarray, sr: int, hop_length: int) -> plt.Figure:
    """Plot AGC gains over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    t = librosa.frames_to_time(np.arange(len(gains)), sr=sr, hop_length=hop_length)
    ax.plot(t, gains)
    ax.axhline(1.0, color="r", linestyle="--", alpha=0.5)
    ax.set(xlabel="Time [sec]", ylabel="Gain", title="AGC Gains")
    plt.tight_layout()
    return fig


def phase_vocoder(
    audio: np.ndarray,
    sr: int,
    stretch_factor: float = 1.5,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    """Time-stretch audio using phase vocoder (preserves pitch)."""
    n_fft = int(window_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)

    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(D), np.angle(D)

    n_frames_out = int(D.shape[1] / stretch_factor)
    out_mag = np.zeros((D.shape[0], n_frames_out))
    out_phase = np.zeros((D.shape[0], n_frames_out))

    freq_bins = np.arange(D.shape[0])
    expected_advance = 2 * np.pi * freq_bins * hop_length / n_fft
    out_phase[:, 0] = phase[:, 0]

    for i in range(n_frames_out):
        pos = i * stretch_factor
        idx = int(pos)
        frac = pos - idx

        if idx + 1 < D.shape[1]:
            out_mag[:, i] = (1 - frac) * mag[:, idx] + frac * mag[:, idx + 1]
            if i > 0:
                diff = phase[:, idx + 1] - phase[:, idx] - expected_advance
                diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
                out_phase[:, i] = out_phase[:, i - 1] + expected_advance + diff
        else:
            out_mag[:, i] = mag[:, -1]
            if i > 0:
                out_phase[:, i] = out_phase[:, i - 1] + expected_advance

    return librosa.istft(
        out_mag * np.exp(1j * out_phase), hop_length=hop_length, n_fft=n_fft
    ).astype(np.float32)


def plot_stretch_comparison(
    orig: np.ndarray, stretched: np.ndarray, sr: int, factor: float
) -> plt.Figure:
    """Compare original and time-stretched audio."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Time Stretch ({factor}x)", fontsize=14)

    n_fft, hop = 320, 160

    for col, (sig, name) in enumerate([(orig, "Original"), (stretched, "Stretched")]):
        t = np.arange(len(sig)) / sr
        axes[0, col].plot(t, sig, linewidth=0.5)
        axes[0, col].set(
            xlabel="Time [sec]",
            ylabel="Amplitude",
            title=f"{name} ({len(sig) / sr:.2f}s)",
        )

        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop)), ref=np.max
        )
        img = librosa.display.specshow(
            S, sr=sr, hop_length=hop, x_axis="time", y_axis="hz", ax=axes[1, col]
        )
        axes[1, col].set(
            xlabel="Time [sec]", ylabel="Frequency [Hz]", title=f"{name} Spectrogram"
        )
        fig.colorbar(img, ax=axes[1, col], format="%+2.0f dB")

    plt.tight_layout()
    return fig


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy.signal.resample."""
    n_samples = int(len(audio) * target_sr / orig_sr)
    return scipy.signal.resample(audio, n_samples).astype(np.float32)


def main():
    audio_path = Path("recording.wav")
    noise_path = Path("stationary_noise.wav")
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    # Q1: Load and resample
    audio_orig, sr_orig = librosa.load(audio_path, sr=None, mono=True, dtype=np.float32)
    audio_32k = resample(audio_orig, int(sr_orig), 32000)

    # Downsample to 16kHz: naive vs proper
    audio_naive = audio_32k[::2].astype(np.float32)
    audio_16k = resample(audio_32k, 32000, 16000)
    sr = 16000

    naive_title = f"Naive Downsample from 32kHz to 16kHz (orig sr={sr_orig}Hz)"
    plot_audio_analysis(audio_naive, sr, naive_title).savefig(out / "q1_naive.png")
    sf.write(out / "q1_naive.wav", audio_naive, sr)

    proper_title = f"Proper Downsample from {sr_orig}Hz to 16kHz"
    plot_audio_analysis(audio_16k, sr, proper_title).savefig(out / "q1_proper.png")
    sf.write(out / "q1_proper.wav", audio_16k, sr)

    # Q2: Add noise
    noise, noise_sr = librosa.load(noise_path, sr=None, mono=True, dtype=np.float32)
    noise_16k = resample(noise, int(noise_sr), 16000)
    min_len = min(len(audio_16k), len(noise_16k))
    noisy = (audio_16k[:min_len] + noise_16k[:min_len]).astype(np.float32)

    plot_signals(
        [
            (audio_16k[:min_len], "Clean"),
            (noise_16k[:min_len], "Noise"),
            (noisy, "Noisy"),
        ],
        sr,
        "Q2: Noise Addition",
    ).savefig(out / "q2_noise.png")
    sf.write(out / "q2_noisy.wav", noisy, sr)

    # Q3: Spectral subtraction
    hop_length = int(10 * sr / 1000)
    enhanced, energy, vad_thresh = spectral_subtraction(noisy, sr)
    plot_vad(energy, vad_thresh, sr, hop_length).savefig(out / "q3_vad.png")
    enhanced_title = "Q3: Spectral Subtraction"
    plot_audio_analysis(enhanced, sr, enhanced_title).savefig(out / "q3_enhanced.png")
    sf.write(out / "q3_enhanced.wav", enhanced, sr)

    # Q4: AGC
    agc_audio, gains = auto_gain_control(audio_16k, sr)
    plot_audio_analysis(agc_audio, sr, "Q4: AGC").savefig(out / "q4_agc.png")
    plot_gains(gains, sr, hop_length).savefig(out / "q4_gains.png")
    sf.write(out / "q4_agc.wav", agc_audio, sr)

    # Q5: Time stretch
    stretched = phase_vocoder(audio_16k, sr, stretch_factor=1.5)
    plot_stretch_comparison(audio_16k, stretched, sr, 1.5).savefig(
        out / "q5_stretch.png"
    )
    sf.write(out / "q5_stretched.wav", stretched, sr)

    print(f"Outputs saved to {out}/")


if __name__ == "__main__":
    main()
