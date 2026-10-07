import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath="UCI_Credit_Card.csv"):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df_clean = df.copy()
    
    # Drop ID column if present
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['ID'])
        
    # Standardize column name PAY_0 to PAY_1 if present
    if 'PAY_0' in df_clean.columns and 'PAY_1' not in df_clean.columns:
        df_clean = df_clean.rename(columns={'PAY_0': 'PAY_1'})
        
    # Standardize target column name
    if 'default.payment.next.month' in df_clean.columns:
        df_clean = df_clean.rename(columns={'default.payment.next.month': 'default'})
        
    # Clean EDUCATION: 0, 5, 6 are undocumented -> map to 4 (Others)
    df_clean['EDUCATION'] = df_clean['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    
    # Clean MARRIAGE: 0 is undocumented -> map to 3 (Others)
    df_clean['MARRIAGE'] = df_clean['MARRIAGE'].replace({0: 3})
    
    return df_clean

def engineer_features(df):
    df_feat = df.copy()
    
    pay_cols = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    # 1. Total & Average Financial Aggregates
    df_feat['BILL_AMT_SUM'] = df_feat[bill_cols].sum(axis=1)
    df_feat['BILL_AMT_AVG'] = df_feat[bill_cols].mean(axis=1)
    df_feat['PAY_AMT_SUM'] = df_feat[pay_amt_cols].sum(axis=1)
    df_feat['PAY_AMT_AVG'] = df_feat[pay_amt_cols].mean(axis=1)
    
    # 2. Credit Utilization Ratios (Bill Amount / Credit Limit)
    for i, col in enumerate(bill_cols, 1):
        df_feat[f'UTIL_{i}'] = df_feat[col] / (df_feat['LIMIT_BAL'] + 1e-5)
    
    util_cols = [f'UTIL_{i}' for i in range(1, 7)]
    df_feat['UTIL_AVG'] = df_feat[util_cols].mean(axis=1)
    df_feat['UTIL_MAX'] = df_feat[util_cols].max(axis=1)
    df_feat['UTIL_MIN'] = df_feat[util_cols].min(axis=1)
    
    # 3. Payment-to-Bill Ratios (Repayment Capability)
    for i in range(1, 6):
        # PAY_AMT_i pays off BILL_AMT_{i+1}
        df_feat[f'PAY_TO_BILL_{i}'] = df_feat[f'PAY_AMT{i}'] / (df_feat[f'BILL_AMT{i+1}'].abs() + 100.0)
    
    pay_bill_cols = [f'PAY_TO_BILL_{i}' for i in range(1, 6)]
    df_feat['PAY_TO_BILL_AVG'] = df_feat[pay_bill_cols].mean(axis=1)
    
    # 4. Payment to Credit Limit Ratio
    df_feat['PAY_TO_LIMIT_RATIO'] = df_feat['PAY_AMT_SUM'] / (df_feat['LIMIT_BAL'] + 1e-5)
    
    # 5. Delinquency Metrics
    df_feat['MAX_DELINQUENCY'] = df_feat[pay_cols].max(axis=1)
    df_feat['AVG_DELINQUENCY'] = df_feat[pay_cols].mean(axis=1)
    # Count of months with payment delays (> 0)
    df_feat['DELINQUENT_MONTHS_CNT'] = (df_feat[pay_cols] > 0).sum(axis=1)
    # Delinquency Trend: Difference between Sept (PAY_1) and Apr (PAY_6)
    df_feat['DELINQUENCY_TREND'] = df_feat['PAY_1'] - df_feat['PAY_6']
    # Severe delinquency flag (late >= 2 months)
    df_feat['SEVERE_DELINQUENCY_FLAG'] = (df_feat[pay_cols] >= 2).any(axis=1).astype(int)
    
    # 6. Unpaid Balance / Debt Accumulation
    for i in range(1, 7):
        df_feat[f'UNPAID_BAL_{i}'] = np.maximum(0, df_feat[f'BILL_AMT{i}'] - df_feat[f'PAY_AMT{i}'])
    unpaid_cols = [f'UNPAID_BAL_{i}' for i in range(1, 7)]
    df_feat['UNPAID_BAL_SUM'] = df_feat[unpaid_cols].sum(axis=1)
    df_feat['UNPAID_BAL_AVG'] = df_feat[unpaid_cols].mean(axis=1)
    
    # 7. Demographic & Credit Ratios
    df_feat['LIMIT_TO_AGE_RATIO'] = df_feat['LIMIT_BAL'] / (df_feat['AGE'] + 1e-5)
    
    return df_feat

def select_model_features(df):
    """
    Select the final feature set validated through
    feature ablation experiments.

    Final feature set:
    - 23 original features
    - Credit Utilization features
    - Payment-to-Bill features
    - Unpaid Balance features
    """

    original_features = [
        'LIMIT_BAL',
        'SEX',
        'EDUCATION',
        'MARRIAGE',
        'AGE',
        'PAY_1',
        'PAY_2',
        'PAY_3',
        'PAY_4',
        'PAY_5',
        'PAY_6',
        'BILL_AMT1',
        'BILL_AMT2',
        'BILL_AMT3',
        'BILL_AMT4',
        'BILL_AMT5',
        'BILL_AMT6',
        'PAY_AMT1',
        'PAY_AMT2',
        'PAY_AMT3',
        'PAY_AMT4',
        'PAY_AMT5',
        'PAY_AMT6'
    ]

    credit_utilization_features = [
        'UTIL_1',
        'UTIL_2',
        'UTIL_3',
        'UTIL_4',
        'UTIL_5',
        'UTIL_6',
        'UTIL_AVG',
        'UTIL_MAX',
        'UTIL_MIN'
    ]

    payment_to_bill_features = [
        'PAY_TO_BILL_1',
        'PAY_TO_BILL_2',
        'PAY_TO_BILL_3',
        'PAY_TO_BILL_4',
        'PAY_TO_BILL_5',
        'PAY_TO_BILL_AVG'
    ]

    unpaid_balance_features = [
        'UNPAID_BAL_1',
        'UNPAID_BAL_2',
        'UNPAID_BAL_3',
        'UNPAID_BAL_4',
        'UNPAID_BAL_5',
        'UNPAID_BAL_6',
        'UNPAID_BAL_SUM',
        'UNPAID_BAL_AVG'
    ]

    final_features = (
        original_features
        + credit_utilization_features
        + payment_to_bill_features
        + unpaid_balance_features
    )

    return df[final_features]

def prepare_train_test_data(filepath="UCI_Credit_Card.csv", test_size=0.2, random_state=42):
    df_raw = load_data(filepath)
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    X = select_model_features(df_feat)
    y = df_feat['default']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, list(X.columns)

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats = prepare_train_test_data()
    print(f"Data Pipeline Ready!")
    print(f"X_train shape: {X_tr.shape}, X_test shape: {X_te.shape}")
    print(f"Total Features: {len(feats)}")
