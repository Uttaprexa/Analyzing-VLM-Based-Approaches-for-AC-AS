# Analyzing VLM-Based Approaches for Anomaly Classification and Segmentation

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](link-to-your-paper)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

> A comprehensive comparative study of WinCLIP and AnomalyCLIP for zero-shot industrial anomaly detection

## Overview

Traditional industrial quality inspection systems require extensive retraining for every new defect type, making them costly and difficult to scale. This project explores how **Vision-Language Models (VLMs)** can revolutionize anomaly detection by enabling zero-shot defect classification through natural language descriptions.

We provide a systematic comparison of two state-of-the-art approaches:
- **WinCLIP**: Window-based dense feature extraction with handcrafted prompts
- **AnomalyCLIP**: Learnable object-agnostic prompts with DPAM attention

### Key Findings

| Method | Classification AUROC | Segmentation AUROC |
|--------|---------------------|-------------------|
| **AnomalyCLIP** | **91.6%** | **90.7%** |
| **WinCLIP** | 61.2% | 72.6% |

AnomalyCLIP demonstrates **49.7% improvement** in classification and **24.9% improvement** in segmentation over WinCLIP.

## Features

- Zero-shot anomaly detection using natural language descriptions
- Comprehensive evaluation on MVTec AD dataset (15 categories, 5,354 images)
- Detailed ablation studies and analysis
- Performance breakdown by object type (texture, rigid, flexible)
- Defect characteristic analysis (contrast, type, size)
- Few-shot learning experiments

## Results

### Classification Performance
![Classification Results](figures/AD.png)

### Segmentation Performance
![Segmentation Results](figures/AS.png)

### Performance by Category Type

| Category Type | AnomalyCLIP | WinCLIP |
|--------------|-------------|---------|
| **Texture Objects** | 98.9% | 56.5% |
| **Rigid Objects** | 87.0% | 68.4% |
| **Flexible Objects** | 87.5% | 55.2% |

## Installation

### Prerequisites
```bash
Python >= 3.8
PyTorch >= 1.10
CUDA >= 11.0 (for GPU support)
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/vlm-anomaly-detection.git
cd vlm-anomaly-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Pretrained Models

**AnomalyCLIP:**
```bash
# Download pretrained weights
wget https://github.com/zqhang/AnomalyCLIP/releases/download/v1.0/anomalyclip_weights.pth
```

**WinCLIP:**
```bash
# WinCLIP uses CLIP pretrained weights (automatically downloaded)
```

## Dataset

We use the **MVTec AD dataset** for evaluation.

### Download MVTec AD
```bash
wget https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz -C ./data/
```

### Dataset Structure
```
data/
├── mvtec_anomaly_detection/
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   └── ...
```

## Usage

### Quick Start - Evaluation

**Evaluate AnomalyCLIP:**
```bash
python evaluate_anomalyclip.py \
    --dataset_path ./data/mvtec_anomaly_detection \
    --checkpoint anomalyclip_weights.pth \
    --output_dir ./results/anomalyclip
```

**Evaluate WinCLIP:**
```bash
python evaluate_winclip.py \
    --dataset_path ./data/mvtec_anomaly_detection \
    --mode zero-shot \
    --output_dir ./results/winclip
```

### Inference on Single Image

```python
from models import AnomalyCLIP
from PIL import Image

# Load model
model = AnomalyCLIP(checkpoint='anomalyclip_weights.pth')

# Load image
image = Image.open('sample_image.jpg')

# Predict
result = model.predict(image)

print(f"Anomaly Score: {result['score']:.4f}")
print(f"Is Anomalous: {result['is_anomalous']}")

# Visualize segmentation
result['segmentation_map'].save('output_segmentation.png')
```

### Few-Shot Learning with WinCLIP

```bash
python evaluate_winclip.py \
    --dataset_path ./data/mvtec_anomaly_detection \
    --mode few-shot \
    --num_shots 4 \
    --output_dir ./results/winclip_fewshot
```

## Reproducing Results

### Full Evaluation Pipeline

Run the complete evaluation pipeline:
```bash
bash scripts/run_full_evaluation.sh
```

This will:
1. Evaluate both models on all 15 MVTec AD categories
2. Generate performance metrics (AUROC, AP, AUPRO)
3. Create visualizations and comparison plots
4. Save results to `./results/`

### Generate Paper Figures

```bash
python scripts/generate_figures.py --results_dir ./results/
```

## Ablation Studies

### Window Size Analysis (WinCLIP)
```bash
python scripts/run_ablation.py \
    --ablation window_size \
    --window_sizes 1 2 3 4 5 6 7
```

### Defect Characteristic Analysis (AnomalyCLIP)
```bash
python scripts/analyze_defect_characteristics.py \
    --results_path ./results/anomalyclip/predictions.json
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{kakda2024analyzing,
  title={Analyzing VLM-Based Approaches for Anomaly Classification and Segmentation: A Comparative Study of WinCLIP and AnomalyCLIP},
  author={Kakda, Mohit and Muthukumaran, Mirudula Shri and Patel, Uttapreksha and Prince, Lawrence Swaminathan Xavier},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Authors

- **Mohit Kakda** - [LinkedIn](your-linkedin) | [GitHub](your-github)
- **Mirudula Shri Muthukumaran** - [LinkedIn](linkedin) | [GitHub](github)
- **Uttapreksha Patel** - [LinkedIn](linkedin) | [GitHub](github)
- **Lawrence Swaminathan Xavier Prince** - [LinkedIn](linkedin) | [GitHub](github)

*Northeastern University, Boston, MA*

## Acknowledgments

- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) - Original AnomalyCLIP implementation
- [WinCLIP](https://github.com/caoyunkang/WinClip) - Original WinCLIP implementation
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) - Benchmark dataset
- [OpenAI CLIP](https://github.com/openai/CLIP) - Foundation vision-language model



---

If you find this project helpful, please consider giving it a star!
