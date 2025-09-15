Enhanced Random Forest for Volcanic Lithology (Huoshiling, Songliao)

This repository contains the reproducible code to compute the metrics reported in the manuscript:

Enhanced Random Forest with Geologically-Informed Feature Optimization for Complex Volcanic Rock Lithology Identification: A Case Study in the Wangfu Fault Depression, Songliao Basin

Task

18-class lithology identification using wire logs (GR, CNL, DEN, AC, RLA5).

Key Metrics Calculated:

Train/Test metrics: Accuracy, Precision, Recall, F1-Score.

Blind-well metrics: Evaluating model generalization on unseen wells.

License

MIT License

1. Environment

To recreate the environment and install dependencies:

Using Conda (recommended):
conda env create -f environment.yml
conda activate erf-volcanic-lithology

Using pip:
# Install dependencies via pip:
pip install -r requirements.txt

2. Usage
Run the Metrics Script

The main script to calculate the metrics is erf_metrics_release.py. You can run it with your own input well log data (in CSV format) as follows:

python erf_metrics_release.py --input /path/to/your/log_data.csv --output /path/to/save/results.csv

Parameters:

input: Path to the input well log CSV file.

output: Path to the output results CSV file.

Example Log File Format:

The CSV file should have the following columns (columns may vary based on data):

GR: Gamma Ray

CNL: Compensated Neutron

DEN: Bulk Density

AC: Acoustic Travel-Time/Sonic

RLA5: Resistivity (5' array)

The logs should include the necessary depth or interval columns, and all logs should be on the same sampling scale.

3. Code Overview
Enhanced Random Forest (eRF)

The eRF model combines Borderline-SMOTE, C4.5 gain-ratio splitting, and Kendall’s coefficient of concordance (Kendall's W) to refine lithology classification. The key workflow steps are:

Borderline-SMOTE is used for class imbalance correction, synthesizing minority instances near decision boundaries.

C4.5 Gain Ratio prioritizes geologically meaningful thresholds for continuous well log features.

Kendall's W quantifies the consistency of feature importance across trees, ensuring that the model relies on stable and robust features.

4. Example Results

Train/test metrics: The accuracy, precision, recall, and F1-score for all classes are provided in the output results.

Blind-well results: The model is also evaluated on blind well data to assess its generalizability.

5. Citation

Please cite this repository and corresponding paper as follows:

Yu T, et al. (2025). Enhanced Random Forest with Geologically-Informed Feature Optimization for Complex Volcanic Rock Lithology Identification: A Case Study in the Wangfu Fault Depression, Songliao Basin. PLOS ONE.

DOI: 10.5281/zenodo.17121939

6. Acknowledgments

This work was supported by the Liaoning Provincial Department of Education (grant no. JYTQN2023207) and the National Natural Science Foundation of China (grant no. 41790453). The funders had no role in the study design, data collection and analysis, decision to publish, or preparation of the manuscript.

7. Contact Information

For any inquiries or issues regarding the code or methodology, please contact:

Taiji Yu
Email: yutj1988@126.com

GitHub: https://github.com/yuzc18

Important Notes:

License: This repository is licensed under the MIT License. You are free to use and modify the code but must provide appropriate attribution to the authors.

Repository Details: Replace the https://github.com/你的用户名 with your actual GitHub username.

Data and Code: This repository assumes all data, except proprietary data, is accessible. It provides a method for users to upload their own well log data for reproducibility. If any additional information about data access or proprietary data needs to be specified, make sure to adjust accordingly.

Code Sharing: The code is publicly available to facilitate reproducibility and reuse, with all necessary instructions for running and modifying the code included in the repository.

This README ensures clarity and completeness, following the guidelines for code sharing and making your repository reproducible for other researchers. Be sure to modify sections such as the GitHub username, data access instructions, and license details to fit your actual setup.
