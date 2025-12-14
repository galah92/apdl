# APDL Assignment 1 - Technical Part

## Setup

```bash
# Merge two recordings (20cm + 3m from mic, truncate 3m to 5s to remove glitch)
ffmpeg -i recording_20cm.wav -i recording_3m.wav -filter_complex "[1:a]atrim=0:5[a1];[0:a][a1]concat=n=2:v=0:a=1" -ac 1 recording.wav -y

# Run
uv sync
uv run main.py
```

## Results

**Q1**: Naive vs proper downsampling. Proper method applies anti-aliasing, preventing frequency folding.

**Q2**: Add stationary noise to clean audio.

**Q3**: Spectral subtraction with VAD-based noise estimation. Pitch contour has gaps during unvoiced/silent segments.

**Q4**: AGC normalizes loud (20cm) and soft (3m) speech to similar levels.

**Q5**: Phase vocoder speeds up 1.5x while preserving pitch (10.16s → 6.77s).
