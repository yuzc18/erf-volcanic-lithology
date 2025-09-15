# Enhanced Random Forest for Volcanic Lithology (Huoshiling, Songliao)

This repository contains the reproducible code to compute the metrics reported in the manuscript:

**Enhanced Random Forest with Geologically-Informed Feature Optimization for Complex Volcanic Rock Lithology Identification: A Case Study in the Wangfu Fault Depression, Songliao Basin**

## Task
- **18-class lithology identification using wire logs** (GR, CNL, DEN, AC, RLA5).

## Key Metrics Calculated:
- **Train/Test metrics**: Accuracy, Precision, Recall, F1-Score.
- **Blind-well metrics**: Evaluating model generalization on unseen wells.

## License
- MIT License

## Data and Code Availability
- **Code**: All author-generated code is available at Zenodo (DOI: [10.5281/zenodo.17121939](https://doi.org/10.5281/zenodo.17121939)) and can be accessed via this [GitHub repository](https://github.com/你的用户名/erf-volcanic-lithology).
- **Data**: Raw well logs cannot be shared due to proprietary reasons. However, users can upload their own well logs in CSV format following the documented schema provided in the repository. Reproducibility is ensured by the provided code and metrics script.

## 1. Environment

To recreate the environment and install dependencies:

- **Using Conda (recommended)**:
```bash
conda env create -f environment.yml
conda activate erf-volcanic-lithology
