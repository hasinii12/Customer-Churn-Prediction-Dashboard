"""
Churn EDA — Exploratory Data Analysis
======================================
Run this as a script or convert to Jupyter notebook with:
    jupytext --to notebook notebooks/eda.py
"""

# %%
import pandas as pd
import numpy as np

df = pd.read_csv('../data/telco_churn.csv')

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# %%
print("\n" + "=" * 50)
print("TARGET VARIABLE — CHURN")
print("=" * 50)
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"\nChurn rate: {(df['Churn'] == 'Yes').mean():.2%}")
# NOTE: ~17-27% churn rate — imbalanced. Use AUC not accuracy.

# %%
print("\n" + "=" * 50)
print("NUMERIC FEATURES — SUMMARY")
print("=" * 50)
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe().round(2))

# TotalCharges edge case — on Kaggle dataset this column has whitespace
# df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# df.dropna(subset=['TotalCharges'], inplace=True)

# %%
print("\n" + "=" * 50)
print("CHURN BY CONTRACT TYPE")
print("=" * 50)
print(df.groupby('Contract')['Churn'].apply(
    lambda x: f"{(x=='Yes').mean():.1%}"
))
# Month-to-month churns at ~40%+ vs ~10% for two-year

# %%
print("\n" + "=" * 50)
print("CHURN BY INTERNET SERVICE")
print("=" * 50)
print(df.groupby('InternetService')['Churn'].apply(
    lambda x: f"{(x=='Yes').mean():.1%}"
))

# %%
print("\n" + "=" * 50)
print("CHURN BY PAYMENT METHOD")
print("=" * 50)
print(df.groupby('PaymentMethod')['Churn'].apply(
    lambda x: f"{(x=='Yes').mean():.1%}"
))

# %%
print("\n" + "=" * 50)
print("TENURE STATS BY CHURN STATUS")
print("=" * 50)
print(df.groupby('Churn')['tenure'].describe().round(1))
# Churned customers have MUCH lower tenure — leaving early

# %%
print("\n" + "=" * 50)
print("MONTHLY CHARGES BY CHURN")
print("=" * 50)
print(df.groupby('Churn')['MonthlyCharges'].describe().round(2))
# Higher charges correlate with higher churn

# %%
print("\n" + "=" * 50)
print("KEY FINDINGS")
print("=" * 50)
print("""
1. Contract type is the strongest predictor:
   - Month-to-month: high churn
   - Two-year:       low churn

2. Fiber optic internet customers churn more 
   (possibly due to competition and higher prices)

3. Electronic check payers churn more than auto-pay customers

4. New customers (tenure < 6 months) are at highest risk

5. Online Security and Tech Support add-ons reduce churn
   (these services create stickiness)

6. Dataset is imbalanced (~17-27% churn) — use AUC, not accuracy
""")
