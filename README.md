# 💳 Credit Scoring & Risk Assessment

A Machine Learning project that predicts the probability of credit card default and assesses customer payment risk.

The project includes data preprocessing, feature engineering, model comparison, threshold optimization, and a Streamlit web application.

## 🚀 Features

- 👤 Single applicant risk assessment
- 📁 Batch applicant scoring using CSV
- ⚡ What-if payment risk analysis
- 📊 Credit score and default probability
- 🔍 Payment history and credit utilization analysis

## 🤖 Machine Learning

Models compared:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Final Model:** CatBoost  
**Features:** 46  
**Decision Threshold:** 0.25

### Evaluation

| Metric | Score |
|---|---:|
| CV ROC-AUC | 0.7876 |
| CV PR-AUC | 0.5617 |
| Test ROC-AUC | 0.7803 |
| Test PR-AUC | 0.5577 |

Recall was given importance because identifying potential defaulters is more important than optimizing accuracy alone.

## ⚙️ Feature Engineering

Created financial behavior features such as:

- Credit utilization
- Payment-to-bill ratios
- Unpaid balances
- Payment history indicators

## 🖥️ Application

The Streamlit application provides:

### Single Applicant
Enter customer financial and payment information to get:

- Credit Score
- Chance of Missing Next Payment
- Payment Risk
- Recommendation
- Key risk factors

### Batch Scoring
Upload a CSV and assess multiple applicants at once.

### What-if Analysis
Change credit usage and payment behavior to understand how the risk situation changes.

## 📂 Project Structure

```text
Credit-Scoring-Model/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── figures/
├── models/
│   └── credit_scoring_model.joblib
│
└── src/
    ├── data_pipeline.py
    ├── model_config.py
    ├── predict.py
    └── train_model.py
🛠️ Tech Stack

Python • Pandas • NumPy • Scikit-learn • CatBoost • XGBoost • LightGBM • Plotly • Streamlit

▶️ Run Locally
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Credit-Scoring-Model

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
📊 Dataset

UCI Credit Card Default dataset.

The dataset is kept locally and is not uploaded to GitHub.

The trained model is provided in:

models/credit_scoring_model.joblib
⚠️ Disclaimer

This is an educational/portfolio project.

The displayed credit score is a project-specific model score and is not an official CIBIL or FICO score.

👨‍💻 Author

Shrirang Ambure

Artificial Intelligence & Data Science