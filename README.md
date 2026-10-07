# 💳 Credit Scoring & Risk Assessment

A Machine Learning project that predicts the probability of credit card default and assesses customer payment risk.

The project includes data preprocessing, feature engineering, model comparison, threshold optimization, and a Streamlit web application for credit risk assessment.

---

## 🚀 Features

- 👤 **Single Applicant Assessment**
- 📁 **Batch Applicant Scoring**
- ⚡ **What-if Risk Analysis**
- 📊 **Credit Score & Default Probability**
- 🔍 **Key Risk Factor Analysis**
- ⚠️ **Payment Risk Assessment**
- 🤖 **Machine Learning Model Comparison**

---

## 🤖 Machine Learning

Five classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### 🏆 Final Model

**CatBoost**

- Features: **46**
- Decision Threshold: **0.25**
- Model selection metric: **ROC-AUC**

### 📈 Model Performance

| Metric | Score |
|---|---:|
| CV ROC-AUC | 0.7876 |
| CV PR-AUC | 0.5617 |
| Test ROC-AUC | 0.7803 |
| Test PR-AUC | 0.5577 |

Recall was given importance because identifying potential defaulters is more important than optimizing accuracy alone.

---

## ⚙️ Feature Engineering

The project creates additional features to capture customer financial behavior.

### 💳 Credit Utilization

Measures how much of the available credit is being used.

### 💰 Payment-to-Bill Ratios

Measures payment behavior relative to outstanding bills.

### 📉 Unpaid Balance

Estimates the amount of bill remaining after payments.

These engineered features are combined with the original customer attributes to create the final **46-feature model input**.

---

## 🖥️ Streamlit Application

The application provides three main sections.

### 👤 Check Applicant

Enter an applicant's financial and payment information to receive:

- 💳 Credit Score
- 📊 Chance of Missing Next Payment
- ⚠️ Payment Risk
- 💡 Recommendation
- 🔍 Key factors affecting the result

### 📁 Check Multiple Applicants

Upload a CSV file to evaluate multiple applicants at once.

The application provides individual predictions and a summary of the results.

### ⚡ What-if Analysis

Change:

- Credit Limit
- Average Monthly Bill
- Payment Delay
- Number of Months With Payment Delay

and see how the payment risk situation changes.

---

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
│   ├── fig1_demographics.png
│   ├── fig2_delinquency_utilization.png
│   └── fig3_model_benchmarks.png
│
├── models/
│   └── credit_scoring_model.joblib
│
└── src/
    ├── data_pipeline.py
    ├── model_config.py
    ├── predict.py
    └── train_model.py
```

---

## 🛠️ Tech Stack

**Python • Pandas • NumPy • Scikit-learn • CatBoost • XGBoost • LightGBM • Plotly • Streamlit**

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Shrirang45/CreditScoringModel.git
cd CreditScoringModel
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Dataset

This project uses the **UCI Credit Card Default dataset**.

The dataset contains customer demographic information, credit limits, payment history, billing amounts, payment amounts, and the target variable indicating whether the customer defaulted on the following month's payment.

The original dataset is kept **locally** and is intentionally excluded from GitHub using `.gitignore`.

---

## 🎯 Risk Threshold

The final application uses a decision threshold of **0.25**.

```text
Default Probability ≥ 0.25
            ↓
       Higher Risk
```

The threshold was selected based on the project's focus on identifying potential defaulters while maintaining a practical balance between precision and recall.

---

## 📈 Model Selection

Models were evaluated using **5-fold stratified cross-validation**.

ROC-AUC was used as the primary model selection metric because it evaluates the model's ability to distinguish between defaulters and non-defaulters across different classification thresholds.

CatBoost achieved the best cross-validation ROC-AUC among the evaluated models and was selected as the final model.

---

## 📁 Important Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `src/data_pipeline.py` | Data cleaning and feature engineering |
| `src/model_config.py` | Final model configuration |
| `src/predict.py` | Production prediction pipeline |
| `src/train_model.py` | Model training and evaluation |
| `models/credit_scoring_model.joblib` | Trained CatBoost model |
| `figures/` | Project analysis and model comparison figures |

---

## ⚠️ Disclaimer

This is an **educational and portfolio project**.

The displayed credit score is a **project-specific model score** and is not an official CIBIL, FICO, or financial institution credit score.

The model should not be used as the sole basis for real-world lending or financial decisions.

---

## 👨‍💻 Author

**Shrirang Ambure**

Artificial Intelligence & Data Science

📍 Pune, Maharashtra

---

⭐ If you find this project useful, consider giving the repository a star!