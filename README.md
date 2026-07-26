# DS_Employee-Attrition-Analysis-and-Prediction
DS_Employee Attrition Analysis and Prediction
# Employee Attrition Analysis and Prediction

A data science project that analyzes employee data to identify key drivers of
attrition and predicts which employees are at risk of leaving, delivered as
an interactive Streamlit dashboard for HR teams.

## Project Structure

```
attrition_project/
├── app.py                     # Streamlit dashboard (main entry point)
├── requirements.txt
├── README.md
├── report.md                  # Approach, results, and key insights
├── data/
│   └── employee_attrition.csv # Source dataset (1470 employees, 35 columns)
├── models/
│   ├── best_model.joblib      # Trained pipeline (preprocessing + classifier)
│   ├── model_comparison.json  # Metrics for all models tried
│   ├── feature_importance.json
│   └── at_risk_employees.csv  # Full workforce scored with attrition probability
└── src/
    ├── preprocessing.py       # Cleaning + feature engineering (shared by training & app)
    └── train_model.py         # Trains and compares models, saves the best one
```

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the model (optional — a trained model is already included)

```bash
cd src
python train_model.py
```

This regenerates `models/best_model.joblib`, `model_comparison.json`,
`feature_importance.json`, and `at_risk_employees.csv`.

### 2. Run the dashboard

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Dashboard Tabs

- **Overview** — headline attrition metrics and departmental breakdown.
- **Explore the Data** — interactive EDA: pick any factor (overtime, income,
  satisfaction, tenure, etc.) and see how it relates to attrition, plus a
  correlation heatmap.
- **Model Performance** — accuracy/precision/recall/F1/AUC-ROC for every
  model tried, confusion matrix, and top features driving attrition.
- **Predict Attrition** — enter a hypothetical or real employee's details
  and get a live attrition risk score.
- **At-Risk Employees** — the full workforce ranked by predicted attrition
  probability, filterable by risk threshold, downloadable as CSV.

## Model

Three classifiers were trained and compared — Logistic Regression, Decision
Tree, and Random Forest — all using `class_weight="balanced"` to account for
the imbalance between employees who left (~16%) and stayed (~84%). The best
model is selected automatically by AUC-ROC and saved as a single scikit-learn
`Pipeline` (preprocessing + classifier together), so the same object handles
raw input end-to-end at prediction time.

See `report.md` for full results and key insights.
