# VSLM Deepfake Detection Benchmarking Pipeline

Benchmark Vision-Small Language Models (VSLMs) on deepfake detection using Zero-Shot and Few-Shot prompting.

## Quick Start

```bash
pip install -r requirements.txt

python scripts/run_zero_shot.py --model qwen_vl --dataset celebdf --max_samples 100

python scripts/run_few_shot.py --model qwen_vl --dataset celebdf --k 2 4 8

python scripts/run_zero_shot.py --dataset celebdf
```

## Supported Models

| Model | Hub ID | VRAM (4-bit) |
|-------|--------|--------------|
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | ~5GB |
| MiniCPM-V 2.6 | `openbmb/MiniCPM-V-2_6` | ~4.5GB |
| MoonDream2 | `vikhyatk/moondream2` | ~2GB |
| InternVL2-4B | `OpenGVLab/InternVL2-4B` | ~3GB |
| LLaVA-NeXT-7B | `llava-hf/llava-v1.6-mistral-7b-hf` | ~5GB |

## Dataset Setup

### Option A: Official Sources (Recommended for Publication)

**CelebDF-v2** (590 real + 5,639 fake videos):
1. Go to: https://github.com/yuezunli/celeb-deepfakeforensics
2. Fill Google Form for access
3. Extract frames to `data/celeb_df/`

**FaceForensics++** (1,000 real + 4,000 manipulated):
1. Go to: https://github.com/ondyari/FaceForensics
2. Fill Google Form for access
3. Use provided download script
4. Extract frames to `data/faceforensics/`

### Option B: Kaggle (Quick Testing)

Kaggle has subsets available for faster setup:
- [CelebDF on Kaggle](https://www.kaggle.com/datasets)
- [FF++ Combined Dataset](https://www.kaggle.com/datasets)

**Note**: Kaggle versions may be incomplete or pre-processed. Use official sources for publishable research.

### Expected Directory Structure

```
data/
├── celeb_df/
│   ├── Celeb-real/
│   │   └── [video_folders]/[frames].jpg
│   └── Celeb-synthesis/
│       └── [video_folders]/[frames].jpg
└── faceforensics/
    ├── original_sequences/youtube/c23/frames/
    └── manipulated_sequences/
        ├── Deepfakes/c23/frames/
        ├── Face2Face/c23/frames/
        ├── FaceSwap/c23/frames/
        └── NeuralTextures/c23/frames/
```

## Project Structure

```
FAKESHEILD/
├── configs/model_configs.yaml
├── src/
│   ├── models/          # VSLM wrappers
│   ├── data/            # Dataset loaders
│   ├── evaluation/      # Zero/Few-shot pipelines
│   └── utils/           # Prompts
├── scripts/             # CLI entry points
└── results/             # Output JSON files
```
