# Analyzing VLM-Based Approaches for Anomaly Classification and Segmentation

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](link-to-your-paper)
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

## Authors

- **Uttapreksha Patel** 
- **Mohit Kakda** 
- **Mirudula Shri Muthukumaran** 
- **Lawrence Swaminathan Xavier Prince** 

*Northeastern University, Boston, MA*

## Acknowledgments

- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) - Original AnomalyCLIP implementation
- [WinCLIP](https://github.com/caoyunkang/WinClip) - Original WinCLIP implementation
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) - Benchmark dataset
- [OpenAI CLIP](https://github.com/openai/CLIP) - Foundation vision-language model

---

If you find this project helpful, please consider giving it a star!
