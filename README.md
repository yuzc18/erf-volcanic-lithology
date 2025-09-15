# Enhanced Random Forest for Volcanic Lithology (Huoshiling, Songliao)

This repository contains the **reproducible code** to compute the metrics reported in the manuscript:

> **Enhanced Random Forest with Geologically-Informed Feature Optimization for Complex Volcanic Rock Lithology Identification: A Case Study in the Wangfu Fault Depression, Songliao Basin.**

- Task: 18-class lithology identification using five logs (GR, CNL, DEN, AC, RLA5).
- This public release **does not** include proprietary well logs. The script calculates:
  - Train/Test set metrics (Accuracy, Precision, Recall, F1).
  - Blind-well metrics.
  - Per-class metrics (one-vs-rest).
- License: MIT.

## 1. Environment

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate erf-volcanic-lithology

# Or pip
pip install -r requirements.txt
