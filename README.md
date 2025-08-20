# Diabetes Risk Prediction and Analysis

## Overview

This is a group project for the Data Mining course at Ho Chi Minh City Open University. The project analyzes the CDC Diabetes Health Indicators dataset to explore relationships between lifestyle, demographic factors, and diabetes risk. We use association rule mining with the Apriori algorithm to discover patterns and machine learning models (Decision Tree, Random Forest, and K-Nearest Neighbors) to predict diabetes status.

Key objectives:
- Explore correlations between lifestyle/health factors (e.g., high blood pressure, cholesterol, physical activity) and diabetes risk.
- Analyze demographic influences (e.g., income, education, gender) on diabetes prevalence.
- Build and evaluate predictive models for early diabetes detection.

Dataset: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) from UCI Machine Learning Repository.
- Samples: 253,680
- Features: 22 (including binary, integer, and categorical variables like BMI, HighBP, Age, Income)

## Installation

### Requirements
- Python 3.8+
- Libraries: 
  ```
  pandas
  numpy
  scikit-learn
  mlxtend
  matplotlib
  seaborn (for visualizations)
  ```

Install dependencies:
```
pip install -r requirements.txt
```
(or manually: `pip install pandas numpy scikit-learn mlxtend matplotlib seaborn`)

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/lymanhthang/Diabetes-Risk-Analysis.git
   cd Diabetes-Risk-Analysis
   ```

2. Download the dataset and place it in the `data/` folder (or use the provided script to fetch it).

3. Run the Jupyter Notebook:
   ```
   jupyter notebook Diabetes_Analysis.ipynb
   ```
   - The notebook covers data preprocessing, EDA, Apriori rule mining, model training, and evaluation.
   - Sections align with the report: preprocessing (Chapter 2), models (Chapter 3), analysis (Chapter 4).

4. For scripts (if separated):
   ```
   python preprocess.py  # Handles cleaning, discretization, SMOTE
   python apriori_analysis.py  # Runs Apriori for rules
   python train_models.py  # Trains and evaluates ML models
   ```

## Project Structure

```
├── data/                  # Dataset files (e.g., diabetes.csv)
├── notebooks/             # Jupyter notebooks for analysis
│   └── Diabetes_Analysis.ipynb
├── scripts/               # Python scripts for modular execution
│   ├── preprocess.py
│   ├── apriori_analysis.py
│   └── train_models.py
├── results/               # Output files (charts, rules, model metrics)
├── report/                # Full project report PDF
│   └── Bao_Cao_KPDL_Nhom_15.pdf
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Key Findings and Results

### Exploratory Data Analysis (EDA)
- High blood pressure, cholesterol, stroke history, heart disease, and low physical activity significantly increase diabetes risk (e.g., people with high BP have higher diabetes rates).
- Correlation heatmap shows strong positive correlations between general health (GenHlth) and physical/mental health issues (0.3–0.52).
- Demographics: Higher income and education levels correlate with lower diabetes risk, especially for females. BMI distribution skews higher in diabetic groups.

### Association Rule Mining (Apriori)
- Parameters: min_support=0.25, min_lift=1.1
- Top rules (lifestyle/demographics → diabetes):
  - {Female} → {not_diabetic} (support=0.487, confidence=0.870, lift=1.011): Females have slightly lower diabetes risk.
  - {Edu_College_Grad} → {not_diabetic} (support=0.382, confidence=0.903, lift=1.049): Higher education linked to better health.
  - {Inc_75k_plus} → {not_diabetic} (support=0.328, confidence=0.920, lift=1.069): High income strongly reduces risk.
  - Rules with high lift emphasize socio-economic factors (e.g., high income + female + non-diabetic → college grad, lift=1.66).

### Machine Learning Models
Models trained on preprocessed data (SMOTE for imbalance, scaling). Evaluated on test set:

| Metric                  | Decision Tree | Random Forest | KNN    |
|-------------------------|---------------|---------------|--------|
| Accuracy                | 0.6518       | 0.7007       | 0.7025 |
| Recall (Diabetic)       | 0.82         | 0.79         | 0.61   |
| F1-Score (Diabetic)    | 0.39         | 0.42         | 0.36   |
| True Positives (TP)    | 5735         | 5521         | 4265   |
| True Negatives (TN)    | 27289        | 30040        | 31342  |

- **Decision Tree:** High recall for diabetic class (82%) but low precision (26%), good for screening but many false positives.
- **Random Forest:** Best overall (accuracy 70.07%), balanced performance, recommended for deployment.
- **KNN:** Similar accuracy but lower recall for diabetic class (61%), misses more cases.

Hyperparameters tuned via GridSearchCV (e.g., max_depth=3-10 for DT, n_estimators=100-200 for RF).

## Contributors
- Lý Mạnh Thắng (2251052112)
- Tôn Quyết Thắng (2251052113)
- Trần Đức Tài (1951052175)

## Limitations and Future Work
- Limitations: Class imbalance affects precision; models may overfit on certain features; dataset lacks temporal data.
- Future Directions: Integrate deep learning (e.g., Neural Networks), add more features (e.g., genetics), deploy as a web app for real-time prediction, or expand to other health datasets.

## References
- UCI Machine Learning Repository: CDC Diabetes Health Indicators.
- [1] Decision Tree: Quinlan, J. R. (1986). Induction of decision trees.
- [2] Random Forest: Breiman, L. (2001). Random forests.
- [3] KNN: Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification.
- [9] SMOTE: Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique.
- [10] Apriori: Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules.
- Full references in the project report PDF.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
